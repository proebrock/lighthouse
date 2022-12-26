import numpy as np
import open3d as o3d



from camsimlib.ray_tracer import RayTracer



class RayTracerEmbree(RayTracer):

    def __init__(self, rays, meshes):
        super(RayTracerEmbree, self).__init__(rays, meshes)



    def run(self):
        """ Run ray tracing
        """
        if self._meshes.num_meshes() == 0:
            return
        # Set up scene
        scene = o3d.t.geometry.RaycastingScene()
        tensor_meshes, _ = self._meshes.to_o3d_tensor_mesh_list()
        geometry_ids = np.zeros(len(tensor_meshes), dtype=int)
        for i, tensor_mesh in enumerate(tensor_meshes):
            # Add_triangles assigns a unique geometry id to each mesh
            # we need to keep this in order to convert it back to an
            # index of our meshlist
            geometry_ids[i] = scene.add_triangles(tensor_mesh)
        rays = self._rays.to_tensor_rays()
        # Run
        result = scene.cast_rays(rays)
        # Extract results and reduce data to valid intersections of rays with triangles
        valid = ~np.isinf(result['t_hit'].numpy())
        self._r.intersection_mask = valid
        self._r.scale = result['t_hit'].numpy()[valid]
        self._r.points_cartesic = self._rays.origs[valid, :] + \
            self._scale[:, np.newaxis] * self._rays.dirs[valid, :]
        self._r.points_barycentric = np.zeros_like(self._r.points_cartesic)
        self._r.points_barycentric[:, 1:] = result['primitive_uvs'].numpy()[valid]
        self._r.points_barycentric[:, 0] = 1.0 - self._r.points_barycentric[:, 1] - \
            self._r.points_barycentric[:, 2]
        self._r.triangle_indices = result['primitive_ids'].numpy()[valid]
        # Convert geometry_ids back to an index inside of our meshlist
        geo_ids = result['geometry_ids'].numpy()[valid]
        sort = np.argsort(geometry_ids)
        rank = np.searchsorted(geometry_ids, geo_ids, sorter=sort)
        self._r.mesh_indices = sort[rank]
        self._r.num_reflections = np.zeros_like(self._r.triangle_indices, dtype=int)
