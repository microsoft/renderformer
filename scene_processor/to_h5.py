import os
import h5py
import numpy as np
import trimesh

from scene_config import SceneConfig


def look_at_to_c2w(camera_position, target_position=[0.0, 0.0, 0.0], up_dir=[0.0, 0.0, 1.0]) -> np.ndarray:
    """
    Look at transform matrix

    :param camera_position: camera position
    :param target_position: target position, default is origin
    :param up_dir: up vector, default is z-axis up
    :return: camera to world matrix
    """

    camera_direction = np.array(camera_position) - np.array(target_position)
    camera_direction = camera_direction / np.linalg.norm(camera_direction)
    camera_right = np.cross(np.array(up_dir), camera_direction)
    camera_right = camera_right / np.linalg.norm(camera_right)
    camera_up = np.cross(camera_direction, camera_right)
    camera_up = camera_up / np.linalg.norm(camera_up)
    rotation_transform = np.zeros((4, 4))
    rotation_transform[0, :3] = camera_right
    rotation_transform[1, :3] = camera_up
    rotation_transform[2, :3] = camera_direction
    rotation_transform[-1, -1] = 1.0
    translation_transform = np.eye(4)
    translation_transform[:3, -1] = -np.array(camera_position)
    look_at_transform = np.matmul(rotation_transform, translation_transform)
    return np.linalg.inv(look_at_transform)


def save_to_h5(scene_config: SceneConfig, mesh_path: str, output_h5_path: str):
    all_triangles = []
    all_vn = []
    all_texture = []

    size = 32
    mask = np.zeros((size, size), dtype=bool)
    x, y = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    mask[x + y <= size] = 1
    
    split_mesh_path = os.path.dirname(mesh_path) + '/split'
    for obj_key, obj_config in scene_config.objects.items():
        mesh: trimesh.Trimesh = trimesh.load(f'{split_mesh_path}/{obj_key}.obj', process=False, force='mesh')  # type: ignore
        triangles = mesh.triangles
        vn = mesh.vertex_normals[mesh.faces]
        
        material_config = obj_config.material
        # Use vertex color as diffuse color
        diffuse = mesh.visual.face_colors[..., :3] / 255.
        specular = np.array(material_config.specular)[None].repeat(triangles.shape[0], axis=0)
        roughness = np.array([material_config.roughness])[None].repeat(triangles.shape[0], axis=0)
        normal = np.zeros_like(diffuse)
        normal[..., 0] = 0.5
        normal[..., 1] = 0.5
        normal[..., 2] = 1.
        irradiance = np.array(material_config.emissive)[None, :].repeat(triangles.shape[0], axis=0)
        texture = np.concatenate([diffuse, specular, roughness, normal, irradiance], axis=1)
        texture = np.repeat(np.repeat(texture[..., None], size, axis=-1)[..., None], size, axis=-1)
        texture[:, :, ~mask] = 0.0

        all_triangles.append(triangles)
        all_vn.append(vn)
        all_texture.append(texture)
    
    all_triangles = np.concatenate(all_triangles, axis=0)
    all_vn = np.concatenate(all_vn, axis=0)
    all_texture = np.concatenate(all_texture, axis=0)
    print(all_texture.shape)
    
    all_c2w = []
    all_fov = []
    for camera_config in scene_config.cameras:
        c2w = look_at_to_c2w(camera_config.position, camera_config.look_at, camera_config.up)
        all_c2w.append(c2w)
        all_fov.append(camera_config.fov)

    all_c2w = np.stack(all_c2w)
    all_fov = np.array(all_fov)

    os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
    with h5py.File(output_h5_path, "w") as f:
        f.create_dataset("triangles", data=all_triangles.astype(np.float32), compression="gzip", compression_opts=9)
        f.create_dataset("vn", data=all_vn.astype(np.float32), compression="gzip", compression_opts=9)
        f.create_dataset("texture", data=all_texture.astype(np.float16), compression="gzip", compression_opts=9)
        f.create_dataset("c2w", data=all_c2w.astype(np.float32), compression="gzip", compression_opts=9)
        f.create_dataset("fov", data=all_fov.astype(np.float32), compression="gzip", compression_opts=9)
