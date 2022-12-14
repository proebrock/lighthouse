from abc import ABC, abstractmethod
import numpy as np
import open3d as o3d



class RayTracer(ABC):

    def __init__(self, rayorigs, raydirs):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays

        Remark: Right now the meshlist or any Open3d object cannot be
        pickled (no support for that in Open3d) which is important for
        RayTracerPython to use multiprocessing; because of this the
        handling of the meshlist has been moved from this base class
        to the derived classes RayTracerPython and RayTracerEmbree.
        """
        # Ray tracer input: rays
        self._rayorigs = np.reshape(np.asarray(rayorigs), (-1, 3))
        self._raydirs = np.reshape(np.asarray(raydirs), (-1, 3))
        # Make sure origs and dirs have same size
        if self._rayorigs.shape[0] == self._raydirs.shape[0]:
            pass
        elif (self._rayorigs.shape[0] == 1) and (self._raydirs.shape[0] > 1):
            n = self._raydirs.shape[0]
            self._rayorigs = np.tile(self._rayorigs, (n, 1))
        elif (self._rayorigs.shape[0] > 1) and (self._raydirs.shape[0] == 1):
            n = self._rayorigs.shape[0]
            self._raydirs = np.tile(self._raydirs, (n, 1))
        else:
            raise ValueError(f'Invalid values for ray origins (shape {self._rayorigs.shape}) and ray directions (shape {self._raydirs.shape})')
        # Ray tracer results
        self._reset_results()



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
