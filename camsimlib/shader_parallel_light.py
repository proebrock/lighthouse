import numpy as np

from camsimlib.shader import Shader



class ShaderParallelLight(Shader):

    def __init__(self, light_direction, max_intensity=1.0):
        super(ShaderParallelLight, self).__init__(max_intensity)
        self._light_direction = np.asarray(light_direction, dtype=float)
        if self._light_direction.ndim != 1 or self._light_direction.size != 3:
            raise Exception(f'Invalid light direction {light_direction}')



    def __str__(self):
        return f'ShaderParallelLight(light_direction={self._light_direction})'



    def get_light_direction(self):
        return self._light_direction



    def run(self, cam, rt_result, mesh):
        # Extract ray tracer results
        P = rt_result.points_cartesic # shape (n, 3)

        # Prepare shader result
        C = np.zeros_like(P)

        # Temporary (?) fix of the incorrect determination of shadow points
        # due to P already lying inside the mesh and the raytracer
        # producing results with scale very close to zero
        triangle_idx = rt_result.triangle_indices
        triangle_normals = mesh.triangle_normals[triangle_idx]
        correction = 1e-3 * triangle_normals

        illu_mask = self._get_illuminated_mask_parallel_light(P + correction, mesh,
        self._light_direction)
        #print(f'Number of points not in shadow {np.sum(illu_mask)}')

        # Extract ray tracer results and mesh elements
        P = P[illu_mask, :] # shape (n, 3)
        Pbary = rt_result.points_barycentric[illu_mask, :] # shape (n, 3)
        triangle_idx = rt_result.triangle_indices[illu_mask] # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = mesh.triangles[triangle_idx, :] # shape (n, 3)
        vertices = mesh.vertices[triangles] # shape (n, 3, 3)
        vertex_normals = mesh.vertex_normals[triangles] # shape (n, 3, 3)

        vertex_intensities = self._get_vertex_intensities_parallel_light( \
            vertex_normals, self._light_direction)  # shape: (n, 3)

        # From vertex intensities determine object colors
        if mesh.has_vertex_colors():
            vertex_colors = mesh.vertex_colors[triangles]
        else:
            vertex_colors = np.ones((triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        object_colors = np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)

        C[illu_mask] = object_colors
        return C
