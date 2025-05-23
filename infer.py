import os
import torch
import h5py
import argparse
import numpy as np
import imageio

from renderformer import RenderFormerRenderingPipeline
from simple_ocio import ToneMapper


def load_single_h5_data(file_path):
    with h5py.File(file_path, 'r') as f:
        triangles = torch.from_numpy(np.array(f['triangles']).astype(np.float32))
        num_tris = triangles.shape[0]
        texture = torch.from_numpy(np.array(f['texture']).astype(np.float32))
        mask = torch.ones(num_tris, dtype=torch.bool)
        vn = torch.from_numpy(np.array(f['vn']).astype(np.float32))
        c2w = torch.from_numpy(np.array(f['c2w']).astype(np.float32))
        fov = torch.from_numpy(np.array(f['fov']).astype(np.float32))

        data = {
            'triangles': triangles,
            'texture': texture,
            'mask': mask,
            'c2w': c2w,
            'fov': fov,
            'vn': vn,
        }
    return data


def main():
    parser = argparse.ArgumentParser(description="Infer using triangle radiosity transformer model")
    parser.add_argument("--h5_file", type=str, required=True, help="Path to the input H5 file")
    parser.add_argument("--model_id", type=str, help="Model ID on Hugging Face or local path", default="renderformer/renderformer-v1.1-swin-large")
    parser.add_argument("--precision", type=str, choices=['bf16', 'fp16', 'fp32'], default='fp16', help="Precision for inference")
    parser.add_argument("--resolution", type=int, default=512, help="Resolution for inference")
    parser.add_argument("--output_dir", type=str, help="Output directory (Default: same as input H5 file)", required=False)
    parser.add_argument("--tone_mapper", type=str, choices=['none', 'agx', 'filmic', 'pbr_neutral'], default='none', help="Tone mapper for inference")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    pipeline = RenderFormerRenderingPipeline.from_pretrained(args.model_id)

    if device == torch.device('cuda') and os.name == 'posix':  # avoid windows
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

    # Load data and move to device
    data = load_single_h5_data(args.h5_file)

    # Add batch dimension to all tensors
    triangles = data['triangles'].unsqueeze(0).to(device)
    texture = data['texture'].unsqueeze(0).to(device)
    mask = data['mask'].unsqueeze(0).to(device)
    vn = data['vn'].unsqueeze(0).to(device)
    c2w = data['c2w'].unsqueeze(0).to(device)
    fov = data['fov'].unsqueeze(0).unsqueeze(-1).to(device)

    rendered_imgs = pipeline(
        triangles=triangles,
        texture=texture,
        mask=mask,
        vn=vn,
        c2w=c2w,
        fov=fov,
        resolution=args.resolution,
        torch_dtype=torch.float16 if args.precision == 'fp16' else torch.bfloat16 if args.precision == 'bf16' else torch.float32,
    )
    print("Inference completed. Rendered images shape:", rendered_imgs.shape)

    output_dir = args.output_dir if args.output_dir else os.path.dirname(args.h5_file)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.h5_file))[0]

    nv = c2w.shape[1]
    for i in range(nv):
        hdr_img = rendered_imgs[0, i].cpu().numpy().astype(np.float32)
        if args.tone_mapper != 'none':
            ldr_img = tone_mapper.hdr_to_ldr(hdr_img)
        else:
            ldr_img = np.clip(hdr_img, 0, 1)
        ldr_img = (ldr_img * 255).astype(np.uint8)

        hdr_path = os.path.join(output_dir, f"{base_name}_view_{i}.exr")
        ldr_path = os.path.join(output_dir, f"{base_name}_view_{i}.png")

        imageio.v3.imwrite(hdr_path, hdr_img)
        imageio.v3.imwrite(ldr_path, ldr_img)

        print(f"Saved {hdr_path} and {ldr_path}")


if __name__ == '__main__':
    main()
