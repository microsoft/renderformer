Run the model-loading regression tests from the repository root after installing
PyTorch and the project dependencies:

```bash
ATTN_IMPL=sdpa HF_HUB_OFFLINE=1 python -m unittest discover -s tests -v
```

These tests use small CPU models and mock Hub downloads. No GPU, model downloads,
or Hugging Face credentials are required. They cover local and remote loading,
configuration and weight round trips, rejection of pickle and corrupt weights,
and preservation of Hub download options. Every test rejects calls to `torch.load`.
