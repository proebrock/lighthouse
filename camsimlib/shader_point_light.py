import numpy as np

from camsimlib.shader import Shader
from camsimlib.mesh_tools import get_points_normals_colors



class ShaderPointLight(Shader):

    def __init__(self, light_position, max_intensity=1.0):
        super(ShaderPointLight, self).__init__(max_intensity)
        self._light_position = np.asarray(light_position, dtype=float)
        if self._light_position.ndim != 1 or self._light_position.size != 3:
            raise Exception(f'Invalid light position {light_position}')



    def __str__(self):
        return f'ShaderPointLight(light_position={self._light_position})'



    def get_light_position(self):
        return self._light_position



    def run(self, cam, rt_result, mesh):
        # Temporary (?) fix of the incorrect determination of shadow points
        # due to P already lying inside the mesh and the raytracer
        # producing results with scale very close to zero
        triangle_idx = rt_result.triangle_indices
        triangle_normals = mesh.triangle_normals[triangle_idx]
        correction = 1e-3 * triangle_normals

        illu_mask = self._get_illuminated_mask_point_light( \
            rt_result.points_cartesic + correction,
            mesh, self._light_position)

        points, normals, colors = get_points_normals_colors( \
            mesh, rt_result, illu_mask)

        intensities = self._get_vertex_intensities_point_light(points,
            normals, self._light_position)

        object_colors = colors * intensities[:, np.newaxis]

        C = np.zeros_like(rt_result.points_cartesic)
        C[illu_mask] = object_colors
        return C
