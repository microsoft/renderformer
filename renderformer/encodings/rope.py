# Modified from https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py

# Copyright (c) 2021 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import log

import torch
from torch.nn import Module
from torch.amp import autocast
from torch import nn, einsum, Tensor

from einops import rearrange, repeat


# rotary embedding helper functions
def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def rotate_half_hf(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@autocast("cuda", enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    dtype = t.dtype

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    # Split t into three parts: left, middle (to be transformed), and right
    t_left = t[..., :start_index]
    t_middle = t[..., start_index:end_index]
    t_right = t[..., end_index:]

    # Apply rotary embeddings without modifying t in place
    t_transformed = (t_middle * freqs.cos() * scale) + (
        rotate_half(t_middle) * freqs.sin() * scale
    )

    out = torch.cat((t_left, t_transformed, t_right), dim=-1)

    return out.type(dtype)


def freqs_to_cos_sin(freqs, scale=1.0, start_index=0, head_dim=None):
    """
    Convert frequencies to cos and sin for rotary embeddings.

    Args:
        freqs (torch.Tensor): The frequency tensor of shape (..., n_freqs).
        scale (float): The scaling factor for the frequencies.
        start_index (int): The starting index of the frequencies.
        head_dim (int): The dimension of the head.
    """
    if head_dim is not None:
        # pad the freqs to match the head_dim
        freqs = freqs[..., : freqs.shape[-1] // 2]
        left_pad = start_index
        right_pad = head_dim // 2 - (left_pad + freqs.shape[-1])
        freqs = torch.cat(
            (torch.zeros((*freqs.shape[:-1], left_pad), device=freqs.device),
             freqs,
             torch.zeros((*freqs.shape[:-1], right_pad), device=freqs.device)),
            dim=-1,
        )
        freqs = torch.cat([freqs, freqs], dim=-1)

    cos = freqs.cos() * scale
    sin = freqs.sin() * scale
    return cos, sin


@autocast("cuda", enabled=False)
def apply_rotary_emb_cossin(q, k, cos, sin):
    """
    q size: (bsz, n_q_head, seq_len, head_dim)
    k size: (bsz, n_kv_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == q.shape[-1]
    ), f"feature dimension {q.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    q = (q * cos) + (
        rotate_half_hf(q) * sin
    )
    k = (k * cos) + (
        rotate_half_hf(k) * sin
    )

    return q.type(dtype), k.type(dtype)


@autocast("cuda", enabled=False)
def apply_rotary_emb_one_cossin(one_tensor, cos, sin):
    """
    one_tensor size: (bsz, n_head, seq_len, head_dim)
    cos size: (bsz, 1, seq_len, head_dim)
    sin size: (bsz, 1, seq_len, head_dim)
    """
    dtype = one_tensor.dtype
    rot_dim = cos.shape[-1]
    assert (
        rot_dim == one_tensor.shape[-1]
    ), f"feature dimension {one_tensor.shape[-1]} is not equal to rotation dimension {rot_dim}"

    # Apply rotary embeddings without modifying t in place
    one_tensor = (one_tensor * cos) + (
        rotate_half_hf(one_tensor) * sin
    )

    return one_tensor.type(dtype)


class TriangleRotaryEmbedding(Module):
    def __init__(
        self,
        dim,
        hf_format=True,
        double_max_freq=False,
    ):
        """
        TriangleRotaryEmbedding is a class that implements the rotary embedding for the triangle.

        Args:
            dim (int): The dimension of the rotary embedding.
            hf_format (bool): Whether to use the huggingface RoPE format.
            double_max_freq (bool): Whether to double the frequency range.
        """
        super().__init__()

        self.hf_format = hf_format

        # log spaced frequencies
        max_freq = log(
            dim // 2 - 1, 2) if not double_max_freq else log(dim - 1, 2)
        freqs = 2 ** torch.linspace(0, max_freq, dim // 2)

        self.freqs = nn.Parameter(freqs, requires_grad=False)

        # dummy for device
        self.register_buffer("dummy", torch.tensor(0), persistent=False)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def get_triangle_freqs(self, pos: Tensor):
        # generate all frequencies for all triangles
        freqs = self.forward(pos)
        freqs = rearrange(
            freqs, "batch n_tris n_verts d -> batch 1 n_tris (n_verts d)"
        )  # 1 for head dim

        if self.hf_format:
            freqs = torch.cat([freqs, freqs], dim=-1)
        else:
            freqs = repeat(freqs, "... f -> ... (f r)", r=2)
        return freqs

    @autocast("cuda", enabled=False)
    def forward(self, t: Tensor, seq_len=None, offset=0):
        freqs = self.freqs
        freqs = einsum("..., f -> ... f", t.type(freqs.dtype), freqs)

        return freqs
