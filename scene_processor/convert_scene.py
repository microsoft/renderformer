import os
import json
import argparse
import tempfile
from dacite import from_dict, Config
from to_h5 import save_to_h5
from scene_mesh import generate_scene_mesh
from scene_config import SceneConfig


def main():
    parser = argparse.ArgumentParser(description="Convert scene config to mesh and h5")
    parser.add_argument("scene_config_path", type=str, help="Path to scene config JSON file")
    parser.add_argument("--mesh_path", type=str, 
                       help="Output path for mesh file. If not provided, a temporary directory will be used", 
                       default=None)
    parser.add_argument("--output_h5_path", type=str,
                       help="Output path for h5 file. If not provided, will use scene_config_path with .h5 extension",
                       default=None)
    args = parser.parse_args()

    with open(args.scene_config_path, 'r') as f:
        config = json.load(f)
    
    scene_config = from_dict(data_class=SceneConfig, data=config, config=Config(check_types=True, strict=True))
    
    scene_config_dir = os.path.dirname(args.scene_config_path)
    
    if args.output_h5_path is None:
        output_h5_path = os.path.splitext(args.scene_config_path)[0] + '.h5'
    else:
        output_h5_path = args.output_h5_path
    
    if args.mesh_path is None:
        print("No mesh path provided, using temporary directory")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_mesh_path = os.path.join(temp_dir, "temp_mesh.obj")
            print(f"Generating mesh in temporary path: {temp_mesh_path}")
            generate_scene_mesh(scene_config, temp_mesh_path, scene_config_dir)
            save_to_h5(scene_config, temp_mesh_path, output_h5_path)
    else:
        print(f"Using provided mesh path: {args.mesh_path}")
        generate_scene_mesh(scene_config, args.mesh_path, scene_config_dir)
        save_to_h5(scene_config, args.mesh_path, output_h5_path)
    
    print(f"Done converting scene config to h5 file: {output_h5_path}")

if __name__ == "__main__":
    main()
