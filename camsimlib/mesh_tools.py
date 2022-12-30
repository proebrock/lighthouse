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
