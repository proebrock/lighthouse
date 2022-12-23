import numpy as np
import open3d as o3d



from camsimlib.ray_tracer import RayTracer



class RayTracerEmbree(RayTracer):

    def __init__(self, rayorigs, raydirs, meshlist):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param meshlist: List of meshes
        """
        super(RayTracerEmbree, self).__init__(rayorigs, raydirs)
        # Ray tracer input: mesh list
        self._meshlist = meshlist
        for mesh in self._meshlist:
            # Make sure each mesh has normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()



    def run(self):
        """ Run ray tracing
        """
        # Reset results
        self._reset_results()
        # Special case: Empty mesh
        if self._meshlist == 0:
            self._intersection_mask = \
                np.zeros(self._rayorigs.shape[0], dtype=bool)
            self._points_cartesic = np.zeros((0, 3))
            self._points_barycentric = np.zeros((0, 3))
            self._triangle_indices = np.zeros(0, dtype=int)
            self._mesh_indices = np.zeros(0, dtype=int)
            self._scale = np.zeros(0)
            return
        # Set up scene
        scene = o3d.t.geometry.RaycastingScene()
        geometry_ids = np.zeros(len(self._meshlist), dtype=int)
        for i, mesh in enumerate(self._meshlist):
            if not mesh.has_triangles():
                continue # Skip empty meshes
            tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            # Add_triangles assigns a unique geometry id to each mesh
            # we need to keep this in order to convert it back to an
            # index of our meshlist
            geometry_ids[i] = scene.add_triangles(tensor_mesh)
        rays = o3d.core.Tensor(np.hstack(( \
            self._rayorigs.astype(np.float32),
            self._raydirs.astype(np.float32))))
        # Run
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
        # Convert geometry_ids back to an index inside of our meshlist
        geo_ids = result['geometry_ids'].numpy()[valid]
        sort = np.argsort(geometry_ids)
        rank = np.searchsorted(geometry_ids, geo_ids, sorter=sort)
        self._mesh_indices = sort[rank]
