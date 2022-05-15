import numpy as np



class ShaderAmbientLight:

    def __init__(self):
        pass



    def __str__(self):
        return f'ShaderAmbientLight()'



    def run(self, cam, ray_tracer, mesh):
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        C = np.ones_like(P)
        return C
