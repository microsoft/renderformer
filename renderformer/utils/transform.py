import torch
from torch.amp import autocast
import roma
from typing import Optional, Tuple


@torch.no_grad()
@autocast("cuda", enabled=False)
def trans_to_cam_coord(c2w: torch.Tensor, triangles: torch.Tensor, vns: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Transform triangles to camera coordinate system.

    Args:
        c2w: (batch_size, 4, 4) tensor of camera to world matrices.
        triangles: (batch_size, n_tris, 3, 3) tensor of triangles.
        vns: (batch_size, n_tris, 3, 3) tensor of vertex

    Returns:
        triangles_cam: (batch_size, n_tris, 3, 3) tensor of triangles in camera coordinate system.
        c2w_cam: (batch_size, 4, 4) tensor of camera to world matrices in camera coordinate system, should always be identity.
        vns_cam: (batch_size, n_tris, 3, 3) tensor of vertex normals in camera coordinate system.
    """
    device = triangles.device
    dtype = triangles.dtype
    T = roma.Rigid.from_homogeneous(c2w)
    T_inv = T.inverse()
    return T_inv[:, None, None].apply(triangles), torch.eye(4, device=device, dtype=dtype).repeat(c2w.shape[0], 1, 1), T_inv[:, None, None].linear_apply(vns) if vns is not None else None
