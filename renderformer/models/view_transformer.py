import torch
import torch.nn as nn

from renderformer.models.config import RenderFormerConfig
from renderformer.encodings.nerf_encoding import NeRFEncoding
from renderformer.layers.attention import TransformerDecoder
from renderformer.layers.dpt import DPTHead

from einops import rearrange


class ViewTransformer(nn.Module):
    def __init__(self, config: RenderFormerConfig):
        super().__init__()
        self.config = config
        if config.pe_type == 'nerf':
            self.pos_pe = NeRFEncoding(
                in_dim=9,
                num_frequencies=config.vertex_pe_num_freqs,
                include_input=True
            )
            self.pe_token_proj = nn.Linear(
                self.pos_pe.get_out_dim(),
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.token_pos_pe_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.token_pos_pe_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
            self.rope_dim = None
        elif config.pe_type == 'rope':
            self.rope_dim = min(config.vertex_pe_num_freqs, config.view_transformer_latent_dim // config.view_transformer_n_heads // 18 * 2)
        else:
            raise ValueError(f"Unsupported positional encoding type: {config.pe_type}")

        self.ray_map_patch_token = nn.Parameter(torch.randn(1, 1, config.view_transformer_latent_dim))
        if config.vdir_pe_type == 'nerf':
            self.vdir_pe = NeRFEncoding(
                in_dim=3,
                num_frequencies=config.vdir_num_freqs,
                include_input=True
            )
            self.ray_map_encoder = nn.Linear(
                self.vdir_pe.get_out_dim() * config.patch_size * config.patch_size,
                config.view_transformer_latent_dim
            )
            if config.norm_type == 'layer_norm':
                self.ray_map_encoder_norm = nn.LayerNorm(config.view_transformer_latent_dim)
            elif config.norm_type == 'rms_norm':
                self.ray_map_encoder_norm = nn.RMSNorm(config.view_transformer_latent_dim)
            else:
                raise ValueError(f"Unsupported normalization type: {config.norm_type}")
        else:
            raise ValueError(f"Unsupported view direction positional encoding type: {config.vdir_pe_type}")

        self.transformer = TransformerDecoder(
            num_layers=self.config.view_transformer_n_layers,
            num_heads=self.config.view_transformer_n_heads,
            hidden_dim=self.config.view_transformer_latent_dim,
            ctx_dim=self.config.latent_dim,
            ffn_hidden_dim=self.config.view_transformer_ffn_hidden_dim,
            dropout=self.config.dropout,
            activation=self.config.activation,
            norm_type=self.config.norm_type,
            norm_first=self.config.norm_first,
            rope_dim=self.rope_dim,
            rope_type=self.config.rope_type,
            rope_double_max_freq=self.config.rope_double_max_freq,
            qk_norm=self.config.qk_norm,
            bias=self.config.bias,
            include_self_attn=self.config.view_transformer_include_self_attn,
            use_swin_attn=self.config.view_transformer_use_swin_attn,
        )
        if not config.use_dpt_decoder:
            self.out_proj = nn.Linear(self.config.view_transformer_latent_dim, self.config.patch_size * self.config.patch_size * (4 if config.include_alpha else 3))
        else:
            self.out_dpt = DPTHead(
                in_channels=self.config.view_transformer_latent_dim,
                features=self.config.dpt_features,
                out_channels=self.config.dpt_out_channels,
                out_dim=4 if config.include_alpha else 3
            )
            self.out_layers = list(range(self.config.view_transformer_n_layers - 4, self.config.view_transformer_n_layers)) if self.config.dpt_out_layers is None else self.config.dpt_out_layers
        self.out_proj_act = nn.ELU(alpha=1e-3)

    def forward(self, camera_o, ray_map, tri_tokens, tri_pos, valid_mask, tf32_mode=False):
        """
        Cross attention between ray map and triangle tokens.

        Args:
            camera_o (torch.Tensor): (B, 3)
            ray_map (torch.Tensor): (B, H, W, 3)
            tri_tokens (torch.Tensor): (B, N_TRIS, D)
            tri_pos (torch.Tensor): (B, N_TRIS, 9)
            valid_mask (torch.Tensor): (B, N_TRIS)
            tf32_mode (bool): whether to use tf32 mode
        Returns:
            decoded_img: (B, 3, H, W)
        """

        # query sequence
        ray_map = self.vdir_pe(ray_map)
        ray_tokens = rearrange(ray_map, 'b (h1 p1) (w1 p2) c -> b (h1 w1) (c p1 p2)', p1=self.config.patch_size, p2=self.config.patch_size)
        patch_h = ray_map.size(1) // self.config.patch_size
        patch_w = ray_map.size(2) // self.config.patch_size
        ray_tokens = self.ray_map_patch_token + self.ray_map_encoder_norm(self.ray_map_encoder(ray_tokens))  # [B, N_PATCHES, D]
        n_patches = ray_tokens.size(1)
        ray_token_pos = camera_o[:, None].repeat(1, n_patches, 3)  # [B, N_PATCHES, 3]

        # positional encoding if use 'nerf' pe
        if self.config.pe_type == 'nerf':
            ray_tokens = ray_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(ray_token_pos)))
            tri_tokens = tri_tokens + self.token_pos_pe_norm(self.pe_token_proj(self.pos_pe(tri_pos)))

        # do per-ray attention
        if self.config.use_dpt_decoder:
            with torch.autocast(device_type="cuda", dtype=torch.float32 if tf32_mode else torch.bfloat16):
                out_features = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, out_layers=self.out_layers, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w)
            decoded_img = self.out_dpt(out_features, patch_h, patch_w, patch_size=self.config.patch_size)
            return self.out_proj_act(decoded_img)
        else:
            seq = self.transformer(ray_tokens, tri_tokens, src_key_padding_mask=valid_mask, triangle_pos=tri_pos, ray_pos=ray_token_pos, tf32_mode=tf32_mode, patch_h=patch_h, patch_w=patch_w)  # [B, N_PATCHES, D]
            decoded_patches = self.out_proj_act(self.out_proj(seq))  # [B, N_PATCHES, P*P*3]
            decoded_img = rearrange(decoded_patches, 'b (h1 w1) (c p1 p2) -> b c (h1 p1) (w1 p2)', p1=self.config.patch_size, p2=self.config.patch_size, h1=patch_h, w1=patch_w)
            return decoded_img
