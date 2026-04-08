# NOTE: install bpy (4.0.0) and bpy_helper (0.0.8) before usage
# pip install bpy==4.0.0 bpy_helper==0.0.8 --extra-index-url https://download.blender.org/pypi/

import bpy
from bpy_helper.camera import create_camera, look_at_to_c2w
from bpy_helper.material import create_specular_roughness_material, create_white_emmissive_material
from bpy_helper.scene import reset_scene, import_3d_model, scene_meshes
from bpy_helper.utils import stdout_redirected
from bpy_helper.io import save_blend_file

import os
import json
import tempfile
import imageio
import numpy as np
from tqdm import tqdm
from dacite import from_dict, Config

BLENDER_BACKEND = os.getenv('BLENDER_BACKEND', 'CUDA')

from scene_config import SceneConfig, CameraConfig
from scene_mesh import generate_scene_mesh


def scene_to_img(
        scene_config: SceneConfig,
        mesh_path: str,
        output_image_path: str,
        save_img: bool = False,
        resolution: int = 512,
        spp: int = 4096,
        dump_blend_file: bool = True,
        skip_rendering: bool = True
    ) -> list[tuple[np.ndarray, np.ndarray]]:


    def setup_blender_scene(scene_config: SceneConfig, mesh_path: str) -> None:
        reset_scene()
        with stdout_redirected():
            split_mesh_path = os.path.dirname(mesh_path) + '/split'
            
            for obj_key, obj_config in scene_config.objects.items():
                import_3d_model(f'{split_mesh_path}/{obj_key}.obj')
                material_config = obj_config.material
                if obj_config.material.emissive[0] > 0:
                    material = create_white_emmissive_material(
                        strength=material_config.emissive[0],
                        material_name=f"{obj_key}"
                    )
                else:
                    material = create_specular_roughness_material(
                        diffuse_color=tuple(material_config.diffuse),
                        specular_color=tuple(material_config.specular),
                        roughness=material_config.roughness,
                        material_name=f"{obj_key}"
                    )
                    if material_config.rand_tri_diffuse_seed:  # Use vertex color as diffuse color when have per-triangle diffuse color
                        bsdf = material.node_tree.nodes["Group"]
                        vcol = material.node_tree.nodes.new(type="ShaderNodeVertexColor")
                        material.node_tree.links.new(vcol.outputs['Color'], bsdf.inputs['Diffuse'])
                
                for obj in scene_meshes():
                    if obj.name == obj_key:
                        obj.data.materials.clear()  # Clear all materials
                        obj.data.materials.append(material)
                        obj.rotation_mode = 'XYZ'
                        obj.rotation_euler = (0.0, 0.0, 0.0)  # Set rotation to (0, 0, 0)

    def render_scene(camera_config: CameraConfig, output_image_path: str, output_obj_path: str) -> tuple[np.ndarray, np.ndarray]:
        camera_pos = np.array(camera_config.position)
        look_at = np.array(camera_config.look_at)
        up = np.array(camera_config.up)
        fov = camera_config.fov
        
        c2w = look_at_to_c2w(camera_pos, look_at, up)
        camera = create_camera(c2w, fov)
        bpy.context.scene.camera = camera
        
        bpy.context.scene.render.resolution_x = resolution
        bpy.context.scene.render.resolution_y = resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.samples = spp
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = BLENDER_BACKEND
        bpy.context.scene.render.threads = 8
        bpy.context.scene.render.threads_mode = 'FIXED'

        bpy.context.scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.  # remove all ambient

        if skip_rendering:
            return np.zeros((resolution, resolution, 4), dtype=np.float32), c2w
        with stdout_redirected():
            temp_img_path = output_image_path.replace(".png", ".exr")
            bpy.context.scene.render.filepath = os.path.abspath(temp_img_path)
            bpy.ops.render.render(animation=False, write_still=True)
            img = imageio.v3.imread(temp_img_path).copy()
            if save_img:
                imageio.v3.imwrite(output_image_path, (img * 255).clip(0, 255).astype(np.uint8))

        return img, c2w

    setup_blender_scene(scene_config, mesh_path)

    results = []
    for i, camera_config in tqdm(list(enumerate(scene_config.cameras))):
        img, c2w = render_scene(camera_config, f"{output_image_path}_{i}.png", f"{output_image_path}_{i}.obj")
        results.append((img, c2w))

    # dump if needed
    if dump_blend_file:
        bpy.ops.file.pack_all()
        save_blend_file(output_image_path + '.blend')

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Render scenes using Blender')
    parser.add_argument('scene_config', type=str, help='Path to scene config JSON file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for rendered images')
    parser.add_argument('--mesh_path', type=str, 
                       help='Path to mesh file. If not provided, a temporary directory will be used',
                       default=None)
    parser.add_argument('--dump_blend', action='store_true', help='Save Blender file after rendering')
    parser.add_argument('--save_img', action='store_true', help='Save rendered images')
    parser.add_argument('--resolution', type=int, default=512, help='Resolution of the rendered images')
    parser.add_argument('--spp', type=int, default=4096, help='Samples per pixel')
    
    args = parser.parse_args()
    
    with open(args.scene_config) as f:
        scene_config = json.load(f)
    scene_config = from_dict(data_class=SceneConfig, data=scene_config, config=Config(check_types=True, strict=True))
        
    os.makedirs(args.output_dir, exist_ok=True)
    output_base = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.scene_config))[0])
    
    if args.mesh_path is None:
        print("No mesh path provided, using temporary directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
            print(f"Generating mesh in temporary path: {temp_mesh_path}")
            generate_scene_mesh(scene_config, temp_mesh_path, os.path.dirname(args.scene_config))
            scene_to_img(
                scene_config=scene_config,
                mesh_path=temp_mesh_path,
                output_image_path=output_base,
                dump_blend_file=args.dump_blend,
                save_img=args.save_img,
                resolution=args.resolution,
                spp=args.spp,
                skip_rendering=False if args.save_img else True
            )
    else:
        print(f"Using provided mesh path: {args.mesh_path}")
        generate_scene_mesh(scene_config, args.mesh_path, os.path.dirname(args.scene_config))
        scene_to_img(
            scene_config=scene_config,
            mesh_path=args.mesh_path,
            output_image_path=output_base,
            dump_blend_file=args.dump_blend,
            save_img=args.save_img,
            resolution=args.resolution,
            spp=args.spp,
            skip_rendering=False if args.save_img else True
        )
