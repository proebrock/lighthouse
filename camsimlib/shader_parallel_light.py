import numpy as np

from camsimlib.shader import Shader
from camsimlib.mesh_tools import get_points_normals_vertices



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
        # Temporary (?) fix of the incorrect determination of shadow points
        # due to P already lying inside the mesh and the raytracer
        # producing results with scale very close to zero
        triangle_idx = rt_result.triangle_indices
        triangle_normals = mesh.triangle_normals[triangle_idx]
        correction = 1e-3 * triangle_normals

        illu_mask = self._get_illuminated_mask_parallel_light( \
            rt_result.points_cartesic + correction,
            mesh, self._light_direction)

        points, normals, colors = get_points_normals_vertices( \
            mesh, rt_result, illu_mask)

        intensities = self._get_vertex_intensities_parallel_light( \
            normals, self._light_direction)

        object_colors = colors * intensities[:, np.newaxis]

        C = np.zeros_like(rt_result.points_cartesic)
        C[illu_mask] = object_colors
        return C
