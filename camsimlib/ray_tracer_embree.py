import numpy as np
import open3d as o3d



from camsimlib.ray_tracer import RayTracer



class RayTracerEmbree(RayTracer):

    def __init__(self, rayorigs, raydirs, vertices, triangles):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param vertices: Vertices, shape (k, 3)
        :param triangles: Triangle indices, shape (l, 3)
        """
        super(RayTracerEmbree, self).__init__(rayorigs, raydirs, vertices, triangles)



    def run(self):
        """ Run ray tracing
        """
        # Reset results
        self._reset_results()
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
