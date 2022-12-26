from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d



from camsimlib.multi_mesh import MultiMesh
from camsimlib.rays import Rays
from camsimlib.ray_tracer_result import RayTracerResult



class RayTracer(ABC):

    def __init__(self, rays, meshes):
        self._rays = rays
        self._meshes = meshes
        self._r = RayTracerResult()



    @abstractmethod
    def run(self):
        pass
