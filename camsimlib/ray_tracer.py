import numpy as np
import multiprocessing



class RayTracer:

    def __init__(self, rayorig, raydirs, triangles):
        """ Intersection of multiple rays with a number of triangles
        :param rayorig: Ray origin, shape (3,)
        :param raydir: Ray direction, shape (n,3), n number of rays
        :param triangles: Triangles, shape (m,3,3), (num triangles, num vertices, XYZ)
        """
        # Ray tracer inputs
        self.rayorig = np.asarray(rayorig)
        self.raydirs = np.asarray(raydirs)
        self.triangles = np.asarray(triangles)
        # Ray tracer results
        self.points_cartesic = None
        self.points_barycentric = None
        self.triangle_indices = None



    def get_points_cartesic(self):
        """ Get intersection points of rays with triangle in Cartesian coordinates (x, y, z)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self.points_cartesic



    def get_points_barycentric(self):
        """ Get intersection points of rays within triangle in barycentric coordinates (1-u-v, u, v)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self.points_barycentric



    def get_triangle_indices(self):
        """ Get indices of triangles intersecting with rays (0..n-1) or -1
        :return: Indices of shape (k,), type int, k number of intersecting rays, k<=m
        """
        return self.triangle_indices



    def ray_mesh_intersect(self, rayindex):
        """ Intersection of a single ray with the triangles of the mesh
        Tests intersection of ray with all triangles and returns the one with lowest Z coordinate
        Based on Möller–Trumbore intersection algorithm (see https://scratchapixel.com)
        """
        # Switch off warnings about divide by zero and invalid float op:
        # We do some batch-computations and check the validity of the results later
        with np.errstate(divide='ignore', invalid='ignore'):
            num_tri = self.triangles.shape[0]
            rays = np.tile(self.raydirs[rayindex], num_tri).reshape((num_tri, 3))
            # Do all calculation no matter if invalid values occur during calculation
            v0 = self.triangles[:, 0, :]
            v0v1 = self.triangles[:, 1, :] - v0
            v0v2 = self.triangles[:, 2, :] - v0
            pvec = np.cross(rays, v0v2, axis=1)
            det = np.sum(np.multiply(v0v1, pvec), axis=1)
            inv_det = 1.0 / det
            tvec = self.rayorig - v0
            u = inv_det * np.sum(np.multiply(tvec, pvec), axis=1)
            qvec = np.cross(tvec, v0v1, axis=1)
            v = inv_det * np.sum(np.multiply(rays, qvec), axis=1)
            t = inv_det * np.sum(np.multiply(v0v2, qvec), axis=1)
            # Check all results for validity
            invalid = np.isclose(det, 0.0)
            invalid = np.logical_or(invalid, u < 0.0)
            invalid = np.logical_or(invalid, u > 1.0)
            invalid = np.logical_or(invalid, v < 0.0)
            invalid = np.logical_or(invalid, (u + v) > 1.0)
            invalid = np.logical_or(invalid, t <= 0.0)
            valid_idx = np.where(~invalid)[0]
            if valid_idx.size == 0:
                # No intersection of ray with any triangle in mesh
                return np.NaN * np.zeros(1+3+3)
            else:
                # triangle_index is the index of the triangle intersection point with
                # the lowest t, which means it is the intersection point closest to the camera
                triangle_index = valid_idx[t[valid_idx].argmin()]
                # Prepare result
                result = np.zeros(1+3+3)
                # Triangle index
                result[0] = triangle_index
                # Cartesic intersection point
                result[1:4] = self.rayorig + self.raydirs[rayindex] * t[triangle_index]
                # Barycentric intersection point
                result[4] = 1.0 - u[triangle_index] - v[triangle_index]
                result[5] = u[triangle_index]
                result[6] = v[triangle_index]
                return result



    def run_serial(self):
        """ Run ray tracing (serial processing)
        """
        # Reset results
        self.points_cartesic = None
        self.points_barycentric = None
        self.triangle_indices = None
        # Prepare results
        result = np.zeros((self.raydirs.shape[0], 1+3+3))
        # Switch off warnings about divide by zero and invalid float op
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(self.raydirs.shape[0]):
                result[i, :] = self.ray_mesh_intersect(i)
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 0])
        self.triangle_indices = result[valid, 0].astype(int)
        self.points_cartesic = result[valid, 1:4]
        self.points_barycentric = result[valid, 4:7]



    def run(self):
        """ Run ray tracing (parallel processing)
        """
        # Reset results
        self.points_cartesic = None
        self.points_barycentric = None
        self.triangle_indices = None
        # Run
        pool = multiprocessing.Pool()
        result_list = pool.map(self.ray_mesh_intersect, range(self.raydirs.shape[0]))
        result = np.asarray(result_list)
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 0])
        self.triangle_indices = result[valid, 0].astype(int)
        self.points_cartesic = result[valid, 1:4]
        self.points_barycentric = result[valid, 4:7]
