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
