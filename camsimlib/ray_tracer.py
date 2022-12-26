from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d


from camsimlib.multi_mesh import MultiMesh


class RayTracer(ABC):

    def __init__(self, rays, meshes):
        self._rays = rays
        self._meshes = meshes



    def get_intersection_mask(self):
        """ Get intersection mask: True for all rays that do intersect
        :return: Intersection mask of shape (m, ), type bool
        """
        return self._intersection_mask



    def get_points_cartesic(self):
        """ Get intersection points of rays with triangle in Cartesian coordinates (x, y, z)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self._points_cartesic



    def get_points_barycentric(self):
        """ Get intersection points of rays within triangle in barycentric coordinates (1-u-v, u, v)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self._points_barycentric



    def get_triangle_indices(self):
        """ Get indices of triangles intersecting with rays (0..n-1) or -1
        :return: Indices of shape (k,), type int, k number of intersecting rays, k<=m
        """
        return self._triangle_indices



    def get_mesh_indices(self):
        """ Get indices of meshes intersecting with rays (0..n-1) or -1
        :return: Indices of shape (k,), type int, k number of intersecting rays, k<=m
        """
        return self._mesh_indices



    def get_scale(self):
        """ Get scale so that "self._rayorigs + self._scale * self._raydirs"
        equals the intersection point
        :return: Scale of shape (k,), k number of intersecting rays, k<=m
        """
        return self._scale



    def _reset_results(self):
        """ Reset raytracer results
        """
        self._intersection_mask = None
        self._points_cartesic = None
        self._points_barycentric = None
        self._triangle_indices = None
        self._mesh_indices = None
        self._scale = None


    @abstractmethod
    def run(self):
        pass
