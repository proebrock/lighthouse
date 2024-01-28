import numpy as np
import open3d as o3d


class Rays:
    """ Class for keeping a number of 3D rays

    Each ray has an origin and a direction. Both are represented by 3D vectors.
    The direction may be a unit vector or have arbitrary length.
    """

    def __init__(self, origs, dirs):
        """ Constructor

        You can provide n ray origins and n ray directions. Or you can
        provide a single ray origin and n ray directions. Or you can
        provide n ray origins and a single ray direction.

        Internally everything is converted internally into n ray origins
        and n ray directions.

        :param origs: Ray origins
        :param dirs: Ray directions
        """
        self.origs = np.reshape(np.asarray(origs), (-1, 3))
        self.dirs = np.reshape(np.asarray(dirs), (-1, 3))
        # Make sure origs and dirs have same size
        if self.origs.shape[0] == self.dirs.shape[0]:
            pass
        elif (self.origs.shape[0] == 1) and (self.dirs.shape[0] > 1):
            n = self.dirs.shape[0]
            self.origs = np.tile(self.origs, (n, 1))
        elif (self.origs.shape[0] > 1) and (self.dirs.shape[0] == 1):
            n = self.origs.shape[0]
            self.dirs = np.tile(self.dirs, (n, 1))
        else:
            raise ValueError(f'Invalid values for ray origins (shape {self.origs.shape}) and ray directions (shape {self.dirs.shape})')



    def __str__(self):
        """ Returns string representation of Rays
        :return: String representation
        """
        return f'Rays({self.origs.shape[0]} rays)'



    def __len__(self):
        """ Returns number of rays
        :return: Number of rays
        """
        return self.origs.shape[0]



    def filter(self, mask):
        """ Returns a new ray object with a limited amount of rays
        :param mask: Mask or indices determining the remaining rays
        :return: Rays object with reduced number of rays
        """
        return Rays(self.origs[mask, :], self.dirs[mask, :])



    def points(self, factors):
        """ Uses factor to convert rays into points
        points = origin + scale * direction
        :param factors: Factors
        :return: Resulting 3D points
        """
        return self.origs + factors[:, np.newaxis] * self.dirs



    def scale(self, factor):
        """ Scale length of direction vectors
        :param factor: Scale factor
        """
        self.dirs *= factor



    def dir_lengths(self):
        """ Calculate ray direction lengths
        :return: Lengths of ray direction vectors
        """
        return np.sqrt(np.sum(np.square(self.dirs), axis=1))



    def normalize(self):
        """ Normalize length of direction vectors (length of 1.0)
        Zero length directions will not be touched.
        """
        dirslen = self.dir_lengths()
        nz_mask = ~np.isclose(dirslen, 0.0)
        self.dirs[nz_mask] /= dirslen[nz_mask, np.newaxis]



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



    def to_points_distances(self, points):
        """ Calculate distances from each of the rays to each of the points provided
        :param points: 3D Points, one for each of the n rays, shape (n, 3)
        :return: Distances
        """
        if points.shape[0] != len(self):
            raise ValueError(f'Provide correct number of points {points.shape[0]} != {len(self)}')
        if points.shape[1] != 3:
            raise ValueError('Provider 3D points')
        d = Rays.__cross(self.dirs, points - self.origs)
        d = np.sqrt(np.sum(np.square(d), axis=1))
        return d / self.dir_lengths()



    def intersect_with_plane(self, plane):
        """ Intersect rays with single plane
        If a ray is perpendicular to plane it does not intersect. The
        intersection mask is returned, shape (n, ) for n rays.
        The scale is the factor by which the ray dir has to be scaled with
        to reach the intersection point on the plane (plus ray orig). The
        scale can be positive which means the intersection point is in the
        direction of the ray. If only those points are wanted, filter with
        scale>0. Shape of scales is (m, ) for m intersecting rays out of
        a total of n rays.
        The intersection points are returned as well, shape (m, 3) for
        m intersecting rays out of a total of n rays.
        Plane equation: All points (x,y,z) are on plane that fulfill the
        equation nx*x + ny*y + nz*z + d = 0 with sqrt(nx**2 + ny**2 + nz**2) == 1
        :param plane: Plane, shape (4, ), see above
        :return: points, mask, scales
        """
        if plane.size != 4:
            raise ValueError('Provide a plane in form (nx, ny, nz, d)')
        n = plane[0:3]
        d = plane[3]
        if not np.isclose(np.linalg.norm(n), 1.0):
            raise ValueError('Plane normal vector has to have length 1')
        a = Rays.__multsum(self.origs, n[np.newaxis, :]) + d
        b = Rays.__multsum(self.dirs, n[np.newaxis, :])
        mask = ~np.isclose(b, 0.0)
        scales = -a[mask] / b[mask]
        points = self.origs[mask] + scales[:, np.newaxis] * self.dirs[mask]
        return points, mask, scales



    def get_mesh(self, color=(0, 0, 0)):
        """ Generate mesh representation of rays as an Open3D LineSet
        :param color:
        """
        line_set = o3d.geometry.LineSet()
        points = np.vstack((self.origs, self.origs + self.dirs))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = np.arange(2 * len(self)).reshape((2, len(self))).T
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)
        return line_set



    def to_o3d_tensors(self):
        """ Convert rays into Open3D tensor object for usage in ray tracing
        :return: Tensor object
        """
        return o3d.core.Tensor(np.hstack(( \
            self.origs.astype(np.float32), \
            self.dirs.astype(np.float32))))
