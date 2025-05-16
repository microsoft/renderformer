from dataclasses import dataclass
from typing import List, Dict, Optional, Literal


@dataclass
class TransformConfig:
    translation: List[float]
    """Translation vector [x, y, z]"""
    rotation: List[float]
    """Rotation angles in degrees [x, y, z]"""
    scale: List[float]
    """Scale factors [x, y, z]"""
    normalize: bool = True
    """Whether to normalize the object to a unit sphere"""


@dataclass
class MaterialConfig:
    diffuse: List[float]
    """Diffuse color of the material"""
    specular: List[float]
    """Specular color of the material"""
    roughness: float
    """Roughness of the material"""
    emissive: List[float]
    """Emissive color of the material"""
    smooth_shading: bool
    """Whether to use smooth shading"""
    rand_tri_diffuse_seed: Optional[int] = None
    """Seed for random triangle diffuse color, if None, use the diffuse color directly"""
    random_diffuse_max: float = 1.0
    """Maximum diffuse color of the material, used for random diffuse color"""
    random_diffuse_type: Literal["per-triangle", "per-shading-group"] = "per-shading-group"
    """Type of random diffuse color assignment, either per triangle or per shading group"""


@dataclass
class ObjectConfig:
    mesh_path: str
    """Relative path to the mesh file"""
    material: MaterialConfig
    """Material of the object"""
    transform: TransformConfig
    """Transform of the object"""
    remesh: bool = False
    """Whether to remesh the object"""
    remesh_target_face_num: int = 2048
    """Target face number of the remeshed object"""


@dataclass
class CameraConfig:
    position: List[float]
    """Position of the camera"""
    look_at: List[float]
    """Look at point of the camera"""
    up: List[float]
    """Up vector of the camera"""
    fov: float
    """Field of view of the camera"""

@dataclass
class SceneConfig:
    scene_name: str
    """Name of the scene"""
    version: str
    """Version of the scene description format"""
    objects: Dict[str, ObjectConfig]
    """Objects in the scene (including lighting)"""
    cameras: List[CameraConfig]
    """Cameras in the scene"""
