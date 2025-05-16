import pymeshlab
import trimesh


def remesh(vertices, faces, target_face_num=3568):
    """
    Process mesh given vertices and faces as numpy arrays.
    
    Args:
        vertices (np.ndarray): Nx3 array of vertex positions
        faces (np.ndarray): Mx3 array of face indices
        target_face_num (int): Target face count for decimation
    
    Returns:
        tuple: (processed_vertices, processed_faces)
    """

    # Create pymeshlab mesh from numpy arrays
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))
    
    # Process with pymeshlab
    ms.meshing_isotropic_explicit_remeshing(
        targetlen=pymeshlab.PercentageValue(0.5),
        featuredeg=30,
        adaptive=False
    )
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_face_num,
        qualitythr=1.0
    )
    
    # Get processed mesh from pymeshlab
    processed_mesh = ms.current_mesh()
    return processed_mesh.vertex_matrix(), processed_mesh.face_matrix()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Remesh a 3D model")
    parser.add_argument("--input", type=str, required=True, help="Path to input mesh file")
    parser.add_argument("--target_face_num", type=int, default=2048, help="Target face count for decimation")
    parser.add_argument("--output", type=str, default=None, help="Path to output mesh file")
    args = parser.parse_args()
    
    mesh = trimesh.load(args.input, process=False, force='mesh')

    vertices, faces = remesh(mesh.vertices, mesh.faces, args.target_face_num)

    # Save processed mesh
    output_path = args.output if args.output else args.input.split(".")[0] + "_remesh.obj"
    processed_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    processed_mesh.export(output_path)
