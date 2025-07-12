import os
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from renderformer.encodings.rope import (
    TriangleRotaryEmbedding,
    freqs_to_cos_sin,
    apply_rotary_emb_cossin,
    apply_rotary_emb_one_cossin
)


EPS = 1e-6

ATTN = os.environ.get('ATTN_IMPL', 'flash_attn')
assert ATTN in ['flash_attn', 'sdpa'], "ATTN_IMPL must be either 'flash_attn' or 'sdpa'"
if ATTN == 'flash_attn':
    try:
        from flash_attn import (
            flash_attn_qkvpacked_func,
            flash_attn_varlen_qkvpacked_func,
            flash_attn_varlen_kvpacked_func
        )
        from flash_attn.bert_padding import pad_input, unpad_input
        from einops import rearrange
    except ImportError:
        print("flash_attn is not installed. Please install it from https://github.com/Dao-AILab/flash-attention.")
        print("Falling back to sdpa.")
        ATTN = 'sdpa'

class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with SwiGLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x))))


class FeedForwardGeLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        """
        Feed forward layer with GeLU activation.
        Args:
            dim (int): input dimension
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout rate, default 0.1
        """
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.dropout(self.w2(self.dropout(F.gelu(self.w1(x)))))


class MultiHeadAttention(nn.Module):
    def __init__(self, query_dim, num_heads, kv_dim=None, bias=True, qk_norm=False, norm_type='layer_norm'):
        super().__init__()
        self.apply_rope_cossin = apply_rotary_emb_cossin

        self.num_heads = num_heads
        self.is_self_attn = kv_dim is None
        kv_dim = query_dim if kv_dim is None else kv_dim

        if self.is_self_attn:
            self.in_proj = nn.Linear(query_dim, 3 * query_dim, bias=bias)
        else:
            self.q_proj = nn.Linear(query_dim, query_dim, bias=bias)
            self.k_proj = nn.Linear(kv_dim, query_dim, bias=bias)
            self.v_proj = nn.Linear(kv_dim, query_dim, bias=bias)
        self.out_proj = nn.Linear(query_dim, query_dim, bias=bias)

        if qk_norm:
            if norm_type == 'layer_norm':
                norm_module = nn.LayerNorm
            elif norm_type == 'rms_norm':
                norm_module = nn.RMSNorm
            else:
                raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")
            self.q_norm = norm_module(query_dim, eps=EPS)
            self.k_norm = norm_module(query_dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, q, k, v, src_key_padding_mask=None, rope_cos=None, rope_sin=None, rope_ctx_cos=None, rope_ctx_sin=None, force_sdpa=False):
        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        bs, src_len = q.shape[0], q.shape[1]
        ctx_len = k.shape[1]

        if self.is_self_attn:
            q, k, v = self.in_proj(q).chunk(3, dim=-1)
        else:
            q = self.q_proj(q)
            k = self.k_proj(k)
            v = self.v_proj(v)

        # qk normalization
        q = self.q_norm(q).type(v.dtype)
        k = self.k_norm(k).type(v.dtype)

        q = q.view(bs, src_len, self.num_heads, -1).transpose(1, 2)  # (bs, num_heads, src_len, head_dim)
        k = k.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)
        v = v.view(bs, ctx_len, self.num_heads, -1).transpose(1, 2)

        # apply rope
        if rope_cos is not None:
            if rope_ctx_cos is None:
                q, k = self.apply_rope_cossin(q, k, rope_cos, rope_sin)
            else:
                q = apply_rotary_emb_one_cossin(q, rope_cos, rope_sin)
                k = apply_rotary_emb_one_cossin(k, rope_ctx_cos, rope_ctx_sin)

        if ATTN == 'sdpa' or force_sdpa:
            # create attention mask
            if src_key_padding_mask is not None:
                assert src_key_padding_mask.shape == (bs, ctx_len), \
                    f"expecting key_padding_mask shape of {(bs, ctx_len)}, but got {src_key_padding_mask.shape}"
                attn_mask = (
                    src_key_padding_mask.view(bs, 1, 1, ctx_len)
                    .expand(-1, self.num_heads, -1, -1)
                    .reshape(bs, self.num_heads, 1, ctx_len)
                )
            else:
                attn_mask = None

            attn_output = F.scaled_dot_product_attention(
                query=q.type(v.dtype),
                key=k.type(v.dtype),
                value=v,
                attn_mask=attn_mask,
            ).transpose(1, 2).contiguous().view(bs, src_len, -1)
        elif ATTN == 'flash_attn':
            # self-attn
            if self.is_self_attn:
                if src_key_padding_mask is not None:
                    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(q.transpose(1, 2), src_key_padding_mask)
                    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k.transpose(1, 2), src_key_padding_mask)
                    v_unpad, indices_v, cu_seqlens_v, max_seqlen_v, _ = unpad_input(v.transpose(1, 2), src_key_padding_mask)
                    qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
                    out_unpad = flash_attn_varlen_qkvpacked_func(
                        qkv_unpad, cu_seqlens_q, max_seqlen_q,
                    )
                    attn_output = pad_input(out_unpad, indices_q, bs, src_len).contiguous().view(bs, src_len, -1)
                else:
                    qkv = torch.stack([
                        q.transpose(1, 2),
                        k.transpose(1, 2),
                        v.transpose(1, 2),
                    ], dim=2)
                    attn_output = flash_attn_qkvpacked_func(qkv).contiguous().view(bs, src_len, -1)
            # cross-attn
            else:
                q_unpad = rearrange(q, "b h s d -> (b s) h d")
                cu_seqlens_q = torch.arange(
                    0, (bs + 1) * src_len, step=src_len, dtype=torch.int32, device=q_unpad.device
                )
                max_seqlen_q = src_len

                k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = unpad_input(k.transpose(1, 2), src_key_padding_mask)
                v_unpad, _, _, _, _ = unpad_input(v.transpose(1, 2), src_key_padding_mask)
                kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)

                out_unpad = flash_attn_varlen_kvpacked_func(
                    q_unpad, kv_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                )
                attn_output = rearrange(
                    out_unpad, "(b s) h d -> b s h d", b=bs
                ).contiguous().view(bs, src_len, -1)
        else:
            raise ValueError("Unsupported attention type. Choose from 'flash_attn' and 'sdpa'.")

        return self.out_proj(attn_output)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


SWIN_ATTN_MASK_CACHE = {}
def get_swin_attn_mask(H, W, window_size, shift_size, device):
    """
    Get the attention mask for Swin Transformer. (Original implementation)
    Args:
        H (int): height of image
        W (int): width of image
        window_size (int): window size
        shift_size (int): shift size
        device (torch.device): device to store the attention mask
    Returns:
        attn_mask: (num_windows, num_windows)
    """
    if (H, W, window_size, shift_size) in SWIN_ATTN_MASK_CACHE:
        return SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)]
    else:
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -window_size),
                    slice(-window_size, -shift_size),
                    slice(-shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = (attn_mask == 0).to(torch.bool)  # nW, window_size * window_size, window_size * window_size
        SWIN_ATTN_MASK_CACHE[(H, W, window_size, shift_size)] = attn_mask
        return attn_mask


class SwinSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size: int = 0,
        bias=True,
        qk_norm=False,
        norm_type='layer_norm'
    ):
        """
        Args:
            dim (int): input dimension
            num_heads (int): number of attention heads
            window_size (int): window size
            shift_size (int): shift size, if None, no shift
            bias (bool): whether to use bias, default True
            qk_norm (bool): whether to normalize query and key, default False
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.in_proj = nn.Linear(dim, 3 * dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)

        if qk_norm:
            if norm_type == 'layer_norm':
                norm_module = nn.LayerNorm
            elif norm_type == 'rms_norm':
                norm_module = nn.RMSNorm
            else:
                raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")
            self.q_norm = norm_module(dim, eps=EPS)
            self.k_norm = norm_module(dim, eps=EPS)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        nW = H * W // self.window_size // self.window_size

        # swin related operations
        if self.shift_size > 0:
            attn_mask = get_swin_attn_mask(H, W, self.window_size, self.shift_size, x.device)  # nW, window_size * window_size, window_size * window_size
            attn_mask = attn_mask.repeat(B, 1, 1)[:, None]  # B * nW, 1, window_size * window_size, window_size * window_size
        else:
            attn_mask = None

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # B * nW, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # B * nW, window_size * window_size, C

        q, k, v = self.in_proj(x_windows).chunk(3, dim=-1)

        # qk normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # (B * nW, num_heads, window_size * window_size, head_dim)
        q = q.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)
        k = k.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)
        v = v.view(B * nW, self.window_size * self.window_size, self.num_heads, -1).transpose(1, 2)

        # apply attention
        attn_output = F.scaled_dot_product_attention(
            query=q.type(v.dtype),
            key=k.type(v.dtype),
            value=v,
            attn_mask=attn_mask,
        ).transpose(1, 2).contiguous().view(B * nW, self.window_size * self.window_size, -1)

        attn_windows = self.out_proj(attn_output)  # B * nW, window_size * window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # reverse cyclic shift
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H W C
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H, W, C)
        return x


class AttentionLayer(nn.Module):
    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        kv_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        bias_kv: bool = False,
        activation: str = 'swiglu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        disable_q_norm: bool = False,
        disable_kv_norm: bool = False,
        qk_norm: bool = False,
        add_self_attn: bool = False,
        use_swin_attn: bool = False,
        window_size: int = 8,
        shift_size: int = 0,
    ):
        """
        Attention layer with feed forward and pre-norm.
        Args:
            query_dim (int): input dimension
            kv_dim (int): key and value dimension, if None, set to query_dim (self-attention)
            num_heads (int): number of attention heads
            hidden_dim (int): feed forward hidden dim
            dropout (float): dropout, default 0.1
            bias (bool): whether to use bias, default True
            bias_kv (bool): whether to use bias for key and value, default False
            activation (str): activation function, choose from 'gelu' and 'swiglu', default 'swiglu'
            norm_type (str): normalization type, choose from 'layer_norm' and 'rms_norm', default 'layer_norm'
            disable_q_norm (bool): disable query normalization, default False
            disable_kv_norm (bool): disable key and value normalization, default False
            qk_norm (bool): whether to apply normalization to query and key, default False
            add_self_attn (bool): whether to add self-attention after cross-attention (cross-attn, self-attn, ffn), default False
            use_swin_attn (bool): whether to use swin self-attention, default False
            window_size (int): window size for swin self-attention, default 8
            shift_size (int): shift size for swin self-attention, default 0 (no shift)
        Returns:
            torch.Tensor: (B, N, query_dim)
        """
        super().__init__()
        self.multihead_attn = MultiHeadAttention(
            query_dim=query_dim,
            num_heads=num_heads,
            kv_dim=kv_dim,
            bias=bias,
            qk_norm=qk_norm,
            norm_type=norm_type
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if bias_kv:
            raise NotImplementedError("Bias for key and value is not supported for now")

        if norm_type == 'layer_norm':
            norm_module = nn.LayerNorm
        elif norm_type == 'rms_norm':
            norm_module = nn.RMSNorm
        else:
            raise ValueError("Unsupported normalization type. Choose from 'layer_norm' and 'rms_norm'.")

        self.query_norm = norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()
        kv_dim = query_dim if kv_dim is None else kv_dim
        if not self.multihead_attn.is_self_attn:
            self.kv_norm = norm_module(kv_dim, eps=EPS) if not disable_kv_norm else nn.Identity()

        self.add_self_attn = add_self_attn
        self.use_swin_attn = use_swin_attn
        if add_self_attn:
            if use_swin_attn:
                self.self_attn = SwinSelfAttention(
                    dim=query_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=shift_size,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type
                )
            else:
                self.self_attn = MultiHeadAttention(
                    query_dim=query_dim,
                    num_heads=num_heads,
                    kv_dim=None,
                    bias=bias,
                    qk_norm=qk_norm,
                    norm_type=norm_type
                )
            self.self_attn_norm = norm_module(query_dim, eps=EPS) if not disable_q_norm else nn.Identity()

        if activation == 'swiglu':
            self.ffn = FeedForwardSwiGLU(
                query_dim,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias
            )
        elif activation == 'gelu':
            self.ffn = FeedForwardGeLU(
                query_dim,
                hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias
            )
        else:
            raise ValueError("Unsupported activation function. Choose from 'gelu' and 'swiglu'.")
        
        self.ffn_norm = norm_module(query_dim, eps=EPS)

    def forward(self, query, kv=None, src_key_padding_mask=None, rope_cos=None, rope_sin=None, rope_ctx_cos=None, rope_ctx_sin=None, force_sdpa=False, patch_h=None, patch_w=None):
        """
        Args:
            query (torch.Tensor): (B, N, query_dim)
            kv (torch.Tensor): (B, N, kv_dim), key and value, if None, set to query (self-attention)
            src_key_padding_mask (torch.Tensor): (B, N), key padding mask, things you want to attend to is True
            rope_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            rope_ctx_cos (torch.Tensor): (B, 1, N, head_dim), cosine tensor for RoPE, None if no RoPE is applied
            rope_ctx_sin (torch.Tensor): (B, 1, N, head_dim), sine tensor for RoPE, None if no RoPE is applied
            patch_h (int): height of the patch, used for swin self-attention
            patch_w (int): width of the patch, used for swin self-attention
        Returns:
            torch.Tensor: (B, N, query_dim)
        """

        bs = query.shape[0]
        src_len = query.shape[1]

        q = self.query_norm(query)
        if self.multihead_attn.is_self_attn:
            kv = q
            ctx_len = src_len
        else:
            kv = self.kv_norm(kv)
            ctx_len = kv.shape[1]

        # multihead attention
        attn_output = self.dropout(self.multihead_attn(q, kv, kv, src_key_padding_mask, rope_cos, rope_sin, rope_ctx_cos, rope_ctx_sin, force_sdpa=force_sdpa))
        query = query + attn_output

        if self.add_self_attn:
            q = self.self_attn_norm(query)
            if self.use_swin_attn:
                q = q.view(bs, patch_h, patch_w, -1)
                self_attn_output = self.self_attn(q)
                self_attn_output = self_attn_output.view(bs, patch_h * patch_w, -1)
            else:
                self_attn_output = self.self_attn(q, q, q, None, rope_cos, rope_sin, force_sdpa=force_sdpa)
            query = query + self.dropout(self_attn_output)

        # feed forward
        query = query + self.dropout(self.ffn(self.ffn_norm(query)))
        return query


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        dropout: float = 0.1,
        bias: bool = True,
        bias_kv: bool = False,
        activation: str = 'gelu',
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        norm_first: bool = True,
        rope_dim: Optional[int] = None,
        rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed'] = 'triangle',
        rope_double_max_freq: bool = False,
        qk_norm: bool = False,
    ):
        super().__init__()
        assert norm_first, "Only support norm_first=True"

        self.head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList([
            AttentionLayer(
                query_dim=hidden_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias,
                bias_kv=bias_kv,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
            ) for _ in range(num_layers)
        ])
        self.rope_dim = rope_dim
        if rope_dim is not None:
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type != 'triangle_mixed':
                assert rope_dim // 2 * 9 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            else:
                print(f"Overriding rope_dim {rope_dim} with {hidden_dim // num_heads} for triangle_mixed")
                rope_dim = hidden_dim // num_heads
            self.rope_emb = TriangleRotaryEmbedding(
                dim=rope_dim,
                double_max_freq=rope_double_max_freq,
            )

    def forward(self, x, src_key_padding_mask=None, triangle_pos=None):
        # src_key_padding_mask: (B, N), key padding mask, things you want to attend to is True
        if self.rope_dim is not None:
            assert triangle_pos is not None, "triangle_pos must be provided if rope_dim is not None"
            rope_freqs = self.rope_emb.get_triangle_freqs(triangle_pos)
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
        else:
            rope_cos = rope_sin = None

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        ffn_hidden_dim: int,
        ctx_dim: Optional[int] = None,
        dropout: float = 0.1,
        include_self_attn: bool = True,
        use_swin_attn: bool = False,
        window_size: int = 8,
        shift_size: int = 4,
        activation: str = 'gelu',
        norm_first: bool = True,
        bias: bool = True,
        bias_kv: bool = False,
        norm_type: Literal['layer_norm', 'rms_norm'] = 'layer_norm',
        qk_norm: bool = False,
        rope_dim: Optional[int] = None,
        rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed'] = 'triangle',
        rope_double_max_freq: bool = False,
    ):
        """
        Transformer decoder. Each layer has cross-attention and self-attention.
        Args:
            num_layers (int): number of decoder layers
            num_heads (int): number of attention heads
            hidden_dim (int): hidden dimension
            ffn_hidden_dim (int): feed forward hidden dimension
            ctx_dim (int): context dimension, if None, set to hidden_dim
            dropout (float): dropout rate, default 0.1
            include_self_attn (bool): whether to include self-attention after cross-attention, default True
            use_swin_attn (bool): whether to use swin self-attention, default False
            activation (str): activation function, default 'gelu'
            norm_first (bool): whether to apply normalization before attention/ffn, default True
            bias (bool): whether to use bias, default True
            bias_kv (bool): whether to use bias for key and value, default False
            norm_type (str): normalization type, choose from 'layer_norm' and 'rms_norm', default 'layer_norm'
            qk_norm (bool): whether to apply normalization to query and key, default False
            rope_dim (int): rotary position embedding dimension, if None, no RoPE is used
            rope_type (str): rotary position embedding type, choose from 'triangle', 'triangle_learned', 'triangle_mixed', default 'triangle'
            rope_double_max_freq (bool): whether to double the max frequency for RoPE, default False
        """
        super().__init__()
        ctx_dim = hidden_dim if ctx_dim is None else ctx_dim
        self.head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList([
            AttentionLayer(
                query_dim=hidden_dim,
                kv_dim=ctx_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                bias=bias,
                bias_kv=bias_kv,
                activation=activation,
                norm_type=norm_type,
                qk_norm=qk_norm,
                add_self_attn=include_self_attn,
                use_swin_attn=use_swin_attn,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else shift_size  # w-attn and swin-attn are on alternate layers
            ) for i in range(num_layers)
        ])

        self.rope_dim = rope_dim
        if rope_dim is not None:
            assert rope_dim % 2 == 0, "rope_dim must be even"
            if rope_type != 'triangle_mixed':
                assert rope_dim // 2 * 9 <= hidden_dim // num_heads, f"rope_dim {rope_dim} is too large for hidden_dim {hidden_dim} and num_heads {num_heads}"
            else:
                print(f"Overriding rope_dim {rope_dim} with {hidden_dim // num_heads} for triangle_mixed")
                rope_dim = hidden_dim // num_heads
            self.rope_emb = TriangleRotaryEmbedding(
                dim=rope_dim,
                double_max_freq=rope_double_max_freq,
            )

    def forward(self, x, ctx, src_key_padding_mask=None, triangle_pos=None, ray_pos=None, out_layers=[], tf32_mode=False, patch_h=None, patch_w=None):
        if self.rope_dim is not None:
            assert triangle_pos is not None and ray_pos is not None, "triangle_pos and ray_pos must be provided if rope_dim is not None"
            rope_freqs = self.rope_emb.get_triangle_freqs(ray_pos)
            rope_cos, rope_sin = freqs_to_cos_sin(rope_freqs, head_dim=self.head_dim)
            rope_ctx_freqs = self.rope_emb.get_triangle_freqs(triangle_pos)
            rope_ctx_cos, rope_ctx_sin = freqs_to_cos_sin(rope_ctx_freqs, head_dim=self.head_dim)
        else:
            rope_cos = rope_sin = rope_ctx_cos = rope_ctx_sin = None

        out_list = []
        for idx, layer in enumerate(self.layers):
            x = layer(x, ctx, src_key_padding_mask=src_key_padding_mask, rope_cos=rope_cos, rope_sin=rope_sin, rope_ctx_cos=rope_ctx_cos, rope_ctx_sin=rope_ctx_sin, force_sdpa=tf32_mode, patch_h=patch_h, patch_w=patch_w)
            if idx in out_layers:
                out_list.append([x])
        return x if not out_list else out_list
