import glob
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
from typing import Optional
import argparse
import imageio
from tqdm import tqdm

from renderformer import RenderFormerRenderingPipeline
from simple_ocio import ToneMapper


class TriangleRenderH5Dataset(Dataset):
    def __init__(self, h5_folder_path: str, padding_length: Optional[int] = None):
        self.file_list = glob.glob(os.path.join(h5_folder_path, '*.h5'))
        self.file_list = natsorted(self.file_list)
        print(f'Found {len(self.file_list)} h5 files in {h5_folder_path}')
        self.padding_length = padding_length

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        with h5py.File(file_path, 'r') as f:
            triangles = torch.from_numpy(np.array(f['triangles'])).float()
            num_tris = triangles.shape[0]
            texture = torch.from_numpy(np.array(f['texture'])).float()
            vn = torch.from_numpy(np.array(f['vn'])).float()
            c2w = np.array(f['c2w']).astype(np.float32)
            fov = np.array(f['fov']).astype(np.float32)

            if self.padding_length is not None:
                triangles = torch.concatenate((triangles, torch.zeros(
                    (self.padding_length - num_tris, *triangles.shape[1:]))), dim=0)
                texture = torch.concatenate((texture, torch.zeros(
                    (self.padding_length - num_tris, *texture.shape[1:]))), dim=0)
                vn = torch.concatenate((vn, torch.zeros(
                    (self.padding_length - num_tris, *vn.shape[1:]))), dim=0)
                mask = torch.zeros(self.padding_length, dtype=torch.bool)
                mask[:num_tris] = True
            else:
                mask = torch.ones(num_tris, dtype=torch.bool)

            data = {
                'triangles': triangles,
                'texture': texture,
                'mask': mask,
                'c2w': torch.from_numpy(c2w).float(),
                'fov': torch.from_numpy(fov).float(),
                'vn': vn,
                'file_path': file_path
            }
        return data


def main():
    parser = argparse.ArgumentParser(description="Batch inference using triangle radiosity transformer model")
    parser.add_argument("--h5_folder", type=str, required=True, help="Path to the folder containing input H5 files")
    parser.add_argument("--model_id", type=str, help="Model ID on Hugging Face or local path", default="renderformer/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='fp16', 
                        help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for inference")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--padding_length", type=int, default=None, help="Padding length for inference")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="Output directory for rendered images (default: same as input folder)")
    parser.add_argument("--save_video", action='store_true', default=True, help="Merge rendered images into a video at video.mp4.")
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='none', help="Tone mapper for inference")
    args = parser.parse_args()

    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load model configuration and weights
    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)

    if device == torch.device('cuda'):
        from renderformer_liger_kernel import apply_kernels
        apply_kernels(pipeline.model)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device == torch.device('mps'):
        args.precision = 'fp32'
        print("bf16 and fp16 will cause too large error in MPS, force using fp32 instead.")

    pipeline.to(device)

    # Tone mapper
    if args.tone_mapper != 'none':
        if args.tone_mapper == 'pbr_neutral':
            args.tone_mapper = 'Khronos PBR Neutral'
        tone_mapper = ToneMapper(args.tone_mapper)
        print(f"Using {args.tone_mapper} tone mapper")

    # Create dataset and dataloader
    dataset = TriangleRenderH5Dataset(args.h5_folder, args.padding_length)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    # Set output directory
    output_dir = args.output_dir if args.output_dir is not None else args.h5_folder
    os.makedirs(output_dir, exist_ok=True)

    if args.save_video:
        video_frames = []

    # Batch inference
    print(f"Starting batch inference on {len(dataset)} files with batch size {args.batch_size}")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        batch_size = batch['triangles'].shape[0]

        # Move data to device
        triangles = batch['triangles'].to(device)
        texture = batch['texture'].to(device)
        mask = batch['mask'].to(device)
        vn = batch['vn'].to(device)
        c2w = batch['c2w'].to(device)
        fov = batch['fov'].unsqueeze(-1).to(device)
        file_paths = batch['file_path']

        # Perform inference
        rendered_imgs = pipeline(
            triangles=triangles,
            texture=texture,
            mask=mask,
            vn=vn,
            c2w=c2w,
            fov=fov,
            resolution=args.resolution,
            torch_dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32
        )

        # Save outputs
        for i in range(batch_size):
            file_path = file_paths[i]
            base_name = os.path.splitext(os.path.basename(file_path))[0]

            nv = c2w.shape[1]
            for view_idx in range(nv):
                hdr_img = rendered_imgs[i, view_idx].cpu().numpy().astype(np.float32)
                if args.tone_mapper != 'none':
                    ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
                else:
                    ldr_img = np.clip(hdr_img, 0, 1)
                ldr_img = (ldr_img * 255).astype(np.uint8)

                hdr_path = os.path.join(output_dir, f"{base_name}_view_{view_idx}.exr")
                ldr_path = os.path.join(output_dir, f"{base_name}_view_{view_idx}.png")

                imageio.v3.imwrite(hdr_path, hdr_img)
                imageio.v3.imwrite(ldr_path, ldr_img)

                if args.save_video:
                    video_frames.append(ldr_img)

    print(f"Output saved to: {output_dir}")

    if args.save_video:
        video_frames = np.array(video_frames)
        video_path = os.path.join(output_dir, 'video.mp4')
        imageio.v3.imwrite(video_path, video_frames, fps=24, quality=9)
        print(f"Video saved to: {video_path}")


if __name__ == '__main__':
    main()
