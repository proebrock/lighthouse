import numpy as np



class ShaderAmbientLight:

    def __init__(self, intensity=0.1):
        self._intensity = intensity



    def __str__(self):
        return f'ShaderAmbientLight(intensity={self._intensity})'



    def run(self, cam, ray_tracer, mesh):
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        C = self._intensity * np.ones_like(P)
        return C
