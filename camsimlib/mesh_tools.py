import numpy as np



def get_triangle_normals(mesh, rt_result, ray_mask):
    tindices = rt_result.triangle_indices[ray_mask]
    return mesh.triangle_normals[tindices]



def get_interpolated_vertex_normals(mesh, rt_result, ray_mask):
    tindices = rt_result.triangle_indices[ray_mask]
    vindices = mesh.triangles[tindices]
    vertex_normals = mesh.vertex_normals[vindices]
    Pbary = rt_result.points_barycentric[ray_mask, :]
    return np.einsum('ijk, ij->ik', vertex_normals, Pbary)



def get_points_normals_vertices(mesh, rt_result, ray_mask):
    tindices = rt_result.triangle_indices[ray_mask]
    vindices = mesh.triangles[tindices]
    points = rt_result.points_cartesic[ray_mask, :]
    Pbary = rt_result.points_barycentric[ray_mask, :]
    # Interpolate normals from vertex normals
    vertex_normals = mesh.vertex_normals[vindices]
    normals = np.einsum('ijk, ij->ik', vertex_normals, Pbary)
    if mesh.vertex_colors.shape[0] > 0:
        # Interpolate colors from vertex colors
        vertex_colors = mesh.vertex_colors[vindices]
        colors = np.einsum('ijk, ij->ik', vertex_colors, Pbary)
    else:
        colors = np.ones((np.sum(ray_mask), 3))
    return points, normals, colors
