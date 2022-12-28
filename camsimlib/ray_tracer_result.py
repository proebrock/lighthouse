import numpy as np



class RayTracerResult:

    def __init__(self, num_rays=0):
        self.clear(num_rays)



    def __str__(self):
        pass



    def isclose(self, other):
        pass



    def clear(self, num_rays=0):
        self.intersection_mask = np.zeros(num_rays, dtype=bool)
        self.points_cartesic = np.zeros((num_rays, 3))
        self.points_barycentric = np.zeros((num_rays, 3))
        self.triangle_indices = np.zeros(num_rays, dtype=int)
        self.mesh_indices = np.zeros(num_rays, dtype=int)
        self.scale = np.zeros(num_rays)
        self.num_reflections = np.zeros(num_rays, dtype=int)



    def reduce_to_valid(self):
        self.points_cartesic = self.points_cartesic[self.intersection_mask]
        self.points_barycentric = self.points_barycentric[self.intersection_mask]
        self.triangle_indices = self.triangle_indices[self.intersection_mask]
        self.mesh_indices = self.mesh_indices[self.intersection_mask]
        self.scale = self.scale[self.intersection_mask]
        self.num_reflections = self.num_reflections[self.intersection_mask]



    def expand(self):
        n = self.intersection_mask.size
        tmp = RayTracerResult(n)
        tmp.intersection_mask = self.intersection_mask
        tmp.points_cartesic[self.intersection_mask] = self.points_cartesic
        tmp.points_barycentric[self.intersection_mask] = self.points_barycentric
        tmp.triangle_indices[self.intersection_mask] = self.triangle_indices
        tmp.mesh_indices[self.intersection_mask] = self.mesh_indices
        tmp.scale[self.intersection_mask] = self.scale
        tmp.num_reflections[self.intersection_mask] = self.num_reflections
        self.__dict__.update(tmp.__dict__)
