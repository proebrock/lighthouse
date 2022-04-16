import numpy as np
import open3d as o3d



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
        self._vertices = vertices
        self._triangles = triangles
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



    def run(self):
        """ Run ray tracing
        """
        # Reset results
        self._intersection_mask = None
        self._points_cartesic = None
        self._points_barycentric = None
        self._triangle_indices = None
        self._scale = None
        # Special case: Empty mesh
        if self._triangles.size == 0:
            self._intersection_mask = \
                np.zeros(self._rayorigs.shape[0], dtype=bool)
            self._points_cartesic = np.zeros((0, 3))
            self._points_barycentric = np.zeros((0, 3))
            self._triangle_indices = np.zeros(0, dtype=int)
            self._scale = np.zeros(0)
            return
        # Run
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self._vertices)
        mesh.triangles = o3d.utility.Vector3iVector(self._triangles)
        mesh.compute_vertex_normals() # necessary?
        mesh.compute_triangle_normals() # necessary?
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(mesh)
        rays = o3d.core.Tensor(np.hstack(( \
            self._rayorigs.astype(np.float32),
            self._raydirs.astype(np.float32))))
        result = scene.cast_rays(rays)
        # Extract results and reduce data to valid intersections of rays with triangles
        valid = ~np.isinf(result['t_hit'].numpy())
        self._intersection_mask = valid
        self._scale = result['t_hit'].numpy()[valid]
        self._points_cartesic = self._rayorigs[valid, :] + \
            self._scale[:, np.newaxis] * self._raydirs[valid, :]
        self._points_barycentric = np.zeros_like(self._points_cartesic)
        self._points_barycentric[:, 1:] = result['primitive_uvs'].numpy()[valid]
        self._points_barycentric[:, 0] = 1.0 - self._points_barycentric[:, 1] - \
            self._points_barycentric[:, 2]
        self._triangle_indices = result['primitive_ids'].numpy()[valid]
