import inspect
import json
import pickle
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from safetensors import SafetensorError
from safetensors.torch import save_model

from renderformer import RenderFormer, RenderFormerRenderingPipeline
from renderformer.models.config import RenderFormerConfig


class PickleMarker:
    """A pickle fixture whose only side effect would be a temporary marker file."""

    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return Path.touch, (self.path,)


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.marker = self.directory / "pickle-executed"
        self.config = RenderFormerConfig(
            latent_dim=24,
            num_layers=1,
            num_heads=2,
            dim_feedforward=32,
            num_register_tokens=2,
            pe_type="nerf",
            vertex_pe_num_freqs=2,
            texture_encode_patch_size=1,
            view_transformer_latent_dim=24,
            view_transformer_ffn_hidden_dim=32,
            view_transformer_n_heads=2,
            view_transformer_n_layers=1,
            patch_size=2,
            use_dpt_decoder=False,
        )
        self.original = RenderFormer(self.config)
        # Official models store a flat dataclass configuration.
        (self.directory / "config.json").write_text(json.dumps(asdict(self.config)))
        save_model(self.original, str(self.directory / "model.safetensors"))

        self.download = Mock(side_effect=self.download_fixture)
        for target in (
            "huggingface_hub.hub_mixin.hf_hub_download",
            "renderformer.models.renderformer.hf_hub_download",
        ):
            download_patch = patch(target, self.download, create=True)
            download_patch.start()
            self.addCleanup(download_patch.stop)
        pickle_patch = patch("torch.load", side_effect=AssertionError("torch.load must never be called"))
        self.pickle_load = pickle_patch.start()
        self.addCleanup(pickle_patch.stop)

    def tearDown(self):
        self.pickle_load.assert_not_called()
        self.assertFalse(self.marker.exists(), "A pickle payload was executed")
        self.assertNotIn("pytorch_model.bin", self.requested_files())

    def download_fixture(self, repo_id, filename, **kwargs):
        self.assertEqual(repo_id, "example/renderformer")
        path = self.directory / filename
        if not path.is_file():
            raise EntryNotFoundError(f"Missing fixture: {filename}")
        return str(path)

    def requested_files(self):
        return [call.kwargs["filename"] for call in self.download.call_args_list]

    def loaders(self):
        for source in (str(self.directory), "example/renderformer"):
            for entry in (RenderFormer, RenderFormerRenderingPipeline):
                yield source, entry

    def assert_loaded(self, loaded):
        model = loaded.model if isinstance(loaded, RenderFormerRenderingPipeline) else loaded
        self.assertEqual(model.config, self.config)
        self.assertFalse(model.training)
        self.assertEqual(model.device, torch.device("cpu"))
        expected, actual = self.original.state_dict(), model.state_dict()
        self.assertEqual(expected.keys(), actual.keys())
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

    def write_pickle(self, filename):
        (self.directory / filename).write_bytes(pickle.dumps(PickleMarker(self.marker)))

    def test_loads_safetensors_through_model_and_pipeline(self):
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                self.assert_loaded(entry.from_pretrained(source))

    def test_safe_weights_are_loaded_even_if_pickle_is_present(self):
        self.write_pickle("pytorch_model.bin")
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                self.assert_loaded(entry.from_pretrained(source))

    def test_rejects_pickle_only_models(self):
        (self.directory / "model.safetensors").unlink()
        self.write_pickle("pytorch_model.bin")
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                    entry.from_pretrained(source)

    def test_rejects_models_without_weights(self):
        (self.directory / "model.safetensors").unlink()
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                with self.assertRaisesRegex(FileNotFoundError, "model.safetensors"):
                    entry.from_pretrained(source)

    def test_rejects_pickle_disguised_as_safetensors(self):
        self.write_pickle("model.safetensors")
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                with self.assertRaises(SafetensorError):
                    entry.from_pretrained(source)

    def test_corrupt_safetensors_never_fall_back_to_pickle(self):
        (self.directory / "model.safetensors").write_bytes(b"invalid safetensors data")
        self.write_pickle("pytorch_model.bin")
        for source, entry in self.loaders():
            with self.subTest(source=source, entry=entry.__name__):
                with self.assertRaises(SafetensorError):
                    entry.from_pretrained(source)

    def test_preserves_hub_download_options(self):
        options = dict(
            revision="test-revision",
            cache_dir=str(self.directory / "cache"),
            force_download=False,
            local_files_only=True,
            token=False,
        )
        if "proxies" in inspect.signature(RenderFormer.from_pretrained).parameters:
            options.update(proxies={"https": "http://proxy.example"}, resume_download=True)
        loaded = RenderFormer.from_pretrained("example/renderformer", strict=True, map_location="cpu", **options)
        self.assert_loaded(loaded)
        self.assertEqual(self.requested_files(), ["config.json", "model.safetensors"])
        for call in self.download.call_args_list:
            self.assertEqual(
                call.kwargs,
                dict(repo_id="example/renderformer", filename=call.kwargs["filename"], **options),
            )

    def test_hub_errors_are_preserved(self):
        response = Mock(headers={}, request=Mock())
        response.json.return_value = {}
        error = HfHubHTTPError("Access denied", response=response)

        def deny_weights(repo_id, filename, **kwargs):
            if filename == "model.safetensors":
                raise error
            return self.download_fixture(repo_id, filename, **kwargs)

        self.download.side_effect = deny_weights
        with self.assertRaises(HfHubHTTPError) as caught:
            RenderFormer.from_pretrained("example/renderformer")
        self.assertIs(caught.exception, error)

    def test_strict_loading_rejects_incompatible_weights(self):
        save_model(torch.nn.Linear(2, 2), str(self.directory / "model.safetensors"))
        with self.assertRaises(RuntimeError):
            RenderFormer.from_pretrained(self.directory, strict=True)

    def test_save_pretrained_round_trip(self):
        self.original.save_pretrained(self.directory)
        self.assert_loaded(RenderFormer.from_pretrained(self.directory))


if __name__ == "__main__":
    unittest.main()
