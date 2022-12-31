from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d



from camsimlib.ray_tracer_result import RayTracerResult



class RayTracer(ABC):

    def __init__(self, rays, mesh):
        self._rays = rays
        self._mesh = mesh
        self.r = RayTracerResult()



    @abstractmethod
    def run(self):
        pass
