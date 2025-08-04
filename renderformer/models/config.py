from dataclasses import dataclass, field
from typing import Literal, List, Optional


@dataclass(frozen=True)
class RenderFormerConfig:
    latent_dim: int = 768
    """The latent dimension of the transformer."""
    num_layers: int = 12
    """The number of layers in the transformer."""
    num_heads: int = 6
    """The number of heads in the transformer."""
    dim_feedforward: int = 768 * 4
    """The number of frequencies in the positional encoding for vertex positions."""
    num_register_tokens: int = 16
    """The dimension of the feedforward network in the transformer."""
    dropout: float = 0.0
    """The dropout rate in the transformer."""
    activation: Literal['gelu', 'swiglu'] = 'swiglu'
    """The activation function in the transformer."""
    norm_type: Literal['layer_norm', 'rms_norm'] = 'rms_norm'
    """The type of normalization to use in the transformer."""
    norm_first: bool = True
    """Whether to normalize the input before the transformer."""
    view_indep_qk_norm: bool = True
    """Whether to apply normalization to query and key."""
    qk_norm: bool = True
    """Whether to apply normalization to query and key."""
    bias: bool = False
    """Whether to use bias in the transformer."""

    pe_type: Literal['nerf', 'rope'] = 'rope'
    """The type of positional encoding to use."""
    rope_type: Literal['triangle', 'triangle_learned', 'triangle_mixed'] = 'triangle'
    """The type of RoPE to use."""
    rope_double_max_freq: bool = False
    """Whether to double the max frequency for RoPE."""
    vertex_pe_num_freqs: int = 12
    """The number of frequencies in the positional encoding for vertex positions."""

    # vertex normal encoder
    use_vn_encoder: bool = True
    """Whether to use the vertex normal encoder."""
    vn_pe_num_freqs: int = 6
    """The number of frequencies in the positional encoding for vertex normals."""
    vn_encoder_norm_type: Literal['none', 'layer_norm', 'rms_norm'] = 'rms_norm'
    """The type of normalization to use in the vertex normal encoder."""

    # texture patch encoder
    texture_encode_patch_size: int = 32
    """The size of the triangle's transformed texture patch for encoder."""
    texture_channels: int = 13  # diffuse, specular, normal, roughness, irradiance
    """The number of channels in the texture patch."""
    texture_encoder_norm_type: Literal['layer_norm', 'rms_norm'] = 'rms_norm'
    """The type of normalization to use in the texture encoder."""

    # view transformer
    view_transformer_latent_dim: int = 768
    """The latent dimension of the view transformer."""
    view_transformer_ffn_hidden_dim: int = 768 * 4
    """The hidden dimension of the feedforward network in the view transformer."""
    view_transformer_n_heads: int = 6
    """The number of heads in the view transformer."""
    view_transformer_n_layers: int = 6
    """The number of layers in the view transformer."""
    view_transformer_include_self_attn: bool = True
    """Whether to include self-attention between ray tokens in the view transformer."""
    view_transformer_use_swin_attn: bool = False
    """Whether to use swin self-attention in the view transformer."""
    vdir_pe_type: Literal['nerf'] = 'nerf'
    """The type of positional encoding to use for view direction."""
    vdir_num_freqs: int = 0
    """The number of frequencies in the positional encoding for view direction."""
    patch_size: int = 8
    """The size of the image patch in the view transformer."""
    include_alpha: bool = False
    """Whether to include the alpha channel in the texture patch."""
    use_dpt_decoder: bool = True
    """Whether to use DPT decoder for rendering."""
    dpt_features: int = 128
    """The dim of internal features in the DPT decoder."""
    dpt_out_channels: List[int] = field(default_factory=lambda: [96, 192, 384, 768])
    """The number of output channels per layer in the DPT decoder."""
    dpt_out_layers: Optional[List[int]] = None
    """The layers to use for the DPT decoder."""
    turn_to_cam_coord: bool = True
    """Whether to turn the triangles to camera coordinate."""
    use_ldr: bool = False
    """Whether to run at LDR mode."""

    def get(self, key, default=None):
        return getattr(self, key, default)
