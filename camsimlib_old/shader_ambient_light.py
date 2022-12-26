import numpy as np

from camsimlib.shader import Shader



class ShaderAmbientLight(Shader):

    def __init__(self, max_intensity=0.1):
        super(ShaderAmbientLight, self).__init__(max_intensity)



    def __str__(self):
        return f'ShaderAmbientLight(intensity={self._max_intensity})'



    def run(self, cam, ray_tracer, mesh):
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        C = self._max_intensity * np.ones_like(P)
        return C
