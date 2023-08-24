import numpy as np

from camsimlib.shader import Shader
from camsimlib.ray_tracer_result import get_interpolated_vertex_colors



class ShaderAmbientLight(Shader):

    def __init__(self, max_intensity=0.1):
        super(ShaderAmbientLight, self).__init__(max_intensity)



    def __str__(self):
        return f'ShaderAmbientLight(intensity={self._max_intensity})'



    def run(self, cam, rt_result, mesh):
        mask = np.ones(rt_result.scale.size, dtype=bool)
        colors = get_interpolated_vertex_colors(mesh, rt_result, mask)
        return colors * self._max_intensity
