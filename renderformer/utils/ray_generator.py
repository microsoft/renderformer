import torch
import torch.nn.functional as F
import torch.nn as nn


class RayGenerator(nn.Module):
    """
    Generate rays from camera model and applies refinement according to image indices.
    """
    def __init__(self):
        super().__init__()

    def forward(
            self,
            c2w,
            fov,
            img_res: int=256
    ):
        """
        Args:
            c2w: (*BATCH_SHAPE, 4, 4)
            fov: (*BATCH_SHAPE, 1)
            img_res: int
        
        Returns:
            rays_o: (*BATCH_SHAPE, 3)  # per batch camera origin, assume perspective camera
            rays_d: (*BATCH_SHAPE, H, W, 3)
        """
        
        batch_shape = c2w.shape[:-2]
        x, y = torch.meshgrid(
            torch.linspace(0.5, img_res - 0.5, img_res, device=c2w.device, dtype=c2w.dtype),
            torch.linspace(0.5, img_res - 0.5, img_res, device=c2w.device, dtype=c2w.dtype),
            indexing='xy')
        cx = cy = img_res / 2
        fx = fy = img_res / 2 / torch.tan(.5 * fov[..., 0, None, None])
        x = x[None].repeat(*batch_shape, 1, 1)
        y = y[None].repeat(*batch_shape, 1, 1)
        dirs = torch.stack([
            (x - cx) / fx,
            -(y - cy) / fy,
            -torch.ones_like(x)
        ], dim=-1)  # [*batch_shape, H, W, 3]
        R = c2w[..., :3, :3]  # [*batch_shape, 3, 3]
        t = c2w[..., :3, 3]  # [*batch_shape, 3]

        rays_d = torch.sum(dirs[..., None, :] * R[..., None, None, :, :], dim=-1)  # [*batch_shape, H, W, 3]
        rays_d = F.normalize(rays_d, dim=-1, p=2)

        return t, rays_d


if __name__ == '__main__':
    ray_generator = RayGenerator()
    bs, V, X = 7, 5, 3
    c2w = torch.randn(bs, V, X, 4, 4)
    fov = torch.randn(bs, V, X, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)

    bs, V = 15, 9
    c2w = torch.randn(bs, V, 4, 4)
    fov = torch.randn(bs, V, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)

    bs = 11
    c2w = torch.randn(bs, 4, 4)
    fov = torch.randn(bs, 1)
    rays_o, rays_d = ray_generator(c2w, fov, 256)
    print(rays_o.shape, rays_d.shape)
