import numpy as np
import multiprocessing



class RayTracer:

    def __init__(self, rayorigs, raydirs, vertices, triangles):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param vertices: Vertices, shape (k, 3)
        :param triangles: Triangle indices, shape (l, 3)
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
        # Ray tracer input: triangles
        self._triangles = np.asarray(vertices)[np.asarray(triangles)]
        # Ray tracer results
        self._intersection_mask = None
        self._points_cartesic = None
        self._points_barycentric = None
        self._triangle_indices = None
        self._scale = None



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



    def get_scale(self):
        """ Get scale so that "self._rayorigs + self._scale * self._raydirs"
        equals the intersection point
        :return: Scale of shape (k,), k number of intersecting rays, k<=m
        """
        return self._scale



    @staticmethod
    def __cross(a, b):
        # faster alternative for np.cross(a, b, axis=1)
        c = np.empty_like(a)
        c[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
        c[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
        c[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        return c



    @staticmethod
    def __multsum(a, b):
        # faster alternative for np.sum(np.multiply(a, b), axis=1)
        c = a * b
        return c[:, 0] + c[:, 1] + c[:, 2]



    def ray_mesh_intersect(self, rayindex):
        """ Intersection of a single ray with the triangles of the mesh
        Tests intersection of ray with all triangles and returns the one with lowest Z coordinate
        Based on Möller–Trumbore intersection algorithm (see https://scratchapixel.com)
        """
        # Switch off warnings about divide by zero and invalid float op:
        # We do some batch-computations and check the validity of the results later
        with np.errstate(divide='ignore', invalid='ignore'):
            num_tri = self._triangles.shape[0]
            rays = np.tile(self._raydirs[rayindex], num_tri).reshape((num_tri, 3))
            # Do all calculation no matter if invalid values occur during calculation
            v0 = self._triangles[:, 0, :]
            v0v1 = self._triangles[:, 1, :] - v0
            v0v2 = self._triangles[:, 2, :] - v0
            pvec = RayTracer.__cross(rays, v0v2)
            det = RayTracer.__multsum(v0v1, pvec)
            inv_det = 1.0 / det
            tvec = self._rayorigs[rayindex] - v0
            u = inv_det * RayTracer.__multsum(tvec, pvec)
            qvec = RayTracer.__cross(tvec, v0v1)
            v = inv_det * RayTracer.__multsum(rays, qvec)
            t = inv_det * RayTracer.__multsum(v0v2, qvec)
            # Check all results for validity
            invalid = np.logical_or.reduce((
                np.isclose(det, 0.0),
                u < 0.0,
                u > 1.0,
                v < 0.0,
                (u + v) > 1.0,
                np.isclose(t, 0.0),
                t < 0.0,
            ))
            valid_idx = np.where(~invalid)[0]
            if valid_idx.size == 0:
                # No intersection of ray with any triangle in mesh
                return np.NaN * np.zeros(3+3+1+1)
            else:
                # triangle_index is the index of the triangle intersection point with
                # the lowest t, which means it is the intersection point closest to the camera
                triangle_index = valid_idx[t[valid_idx].argmin()]
                # Prepare result
                result = np.zeros(3+3+1+1)
                # Cartesic intersection point
                result[0:3] = self._rayorigs[rayindex] + self._raydirs[rayindex] * t[triangle_index]
                # Barycentric intersection point
                result[3] = 1.0 - u[triangle_index] - v[triangle_index]
                result[4] = u[triangle_index]
                result[5] = v[triangle_index]
                # Triangle index
                result[6] = triangle_index
                # Scale
                result[7] = t[triangle_index]
                return result



    def run_serial(self):
        """ Run ray tracing (serial processing)
        Is useful for profiling the software without the problem of having multiple processes
        """
        # Reset results
        self._intersection_mask = None
        self._points_cartesic = None
        self._points_barycentric = None
        self._triangle_indices = None
        self._scale = None
        # Prepare results
        result = np.zeros((self._raydirs.shape[0], 3+3+1+1))
        # Switch off warnings about divide by zero and invalid float op
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(self._raydirs.shape[0]):
                result[i, :] = self.ray_mesh_intersect(i)
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 6])
        self._intersection_mask = valid
        self._points_cartesic = result[valid, 0:3]
        self._points_barycentric = result[valid, 3:6]
        self._triangle_indices = result[valid, 6].astype(int)
        self._scale = result[valid, 7]



    def run(self):
        """ Run ray tracing (parallel processing)
        """
        # Reset results
        self._intersection_mask = None
        self._points_cartesic = None
        self._points_barycentric = None
        self._triangle_indices = None
        self._scale = None
        # Run
        pool = multiprocessing.Pool()
        result_list = pool.map(self.ray_mesh_intersect, range(self._raydirs.shape[0]))
        result = np.asarray(result_list)
        # Extract results and reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 6])
        self._intersection_mask = valid
        self._points_cartesic = result[valid, 0:3]
        self._points_barycentric = result[valid, 3:6]
        self._triangle_indices = result[valid, 6].astype(int)
        self._scale = result[valid, 7]
