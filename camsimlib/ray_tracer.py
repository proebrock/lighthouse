import numpy as np



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



    def __ray_mesh_intersect(self, raydir):
        """ Intersection of an ray with a number of triangles
        Tests intersection of ray with all triangles and returns the one with lowest Z coordinate
        Based on Möller–Trumbore intersection algorithm (see https://scratchapixel.com)
        """
        num_tri = self.triangles.shape[0]
        rays = np.tile(raydir, num_tri).reshape((num_tri, 3))
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
            return np.NaN * np.zeros(3), np.NaN * np.zeros(3), -1
        # triangle_index is the index of the triangle intersection point with
        # the lowest t, which means it is the intersection point closest to the camera
        triangle_index = valid_idx[t[valid_idx].argmin()]
        P = self.rayorig + raydir * t[triangle_index]
        Pbary = np.array([
            1.0 - u[triangle_index] - v[triangle_index],
            u[triangle_index], v[triangle_index]])
        return P, Pbary, triangle_index



    def run(self):
        """ Run ray tracing
        """
        # Reset results
        self.points_cartesic = None
        self.points_barycentric = None
        self.triangle_indices = None
        # Switch off warnings about divide by zero and invalid float op
        P = np.zeros_like(self.raydirs)
        Pbary = np.zeros_like(self.raydirs)
        triangle_idx = np.zeros(self.raydirs.shape[0], dtype=int)
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(self.raydirs.shape[0]):
                P[i, :], Pbary[i, :], triangle_idx[i] = \
                    self.__ray_mesh_intersect(self.raydirs[i, :])
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(P[:, 0])
        self.points_cartesic = P[valid, :]
        self.points_barycentric = Pbary[valid, :]
        self.triangle_indices = triangle_idx[valid]
