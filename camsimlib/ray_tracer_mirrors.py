import numpy as np

from camsimlib.ray_tracer import RayTracer as RayTracerBaseClass
from camsimlib.ray_tracer_embree import RayTracerEmbree as RayTracer



class RayTracerMirrors(RayTracerBaseClass):

    def __init__(self, rayorigs, raydirs, meshlist, mirrors=None):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param meshlist: List of meshes
        :param mirrors: List of meshes that are mirrors, same length as meshlist;
            if not provided, no meshes in meshlist are considered mirrors
        """
        super(RayTracerMirrors, self).__init__(rayorigs, raydirs)
        # Ray tracer input: mesh list
        self._meshlist = meshlist
        for mesh in self._meshlist:
            # Make sure each mesh has normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
        # Flags that determine what meshes are mirrors
        if mirrors is None:
            self._mirrors = np.zeros(len(meshlist), dtype=bool)
        else:
            self._mirrors = np.asarray(mirrors, dtype=bool)
            if len(meshlist) != self._mirrors.size:
                raise Exception('Provide proper mirror flags for meshes')
        # Maximum number  of reflections before ray tracing is aborted
        self._max_num_reflections = 10



    def get_num_reflections(self):
        return self._num_reflections



    @staticmethod
    def __mirror_vector(vecs, normals):
        assert vecs.shape == normals.shape
        vn = np.sum(vecs * normals, axis=1)
        return vecs - 2 * vn[:, np.newaxis] * normals



    def run(self):
        # Special case: Empty mesh list
        if len(self._meshlist) == 0:
            self._intersection_mask = \
                np.zeros(self._rayorigs.shape[0], dtype=bool)
            self._points_cartesic = np.zeros((0, 3))
            self._points_barycentric = np.zeros((0, 3))
            self._triangle_indices = np.zeros(0, dtype=int)
            self._mesh_indices = np.zeros(0, dtype=int)
            self._scale = np.zeros(0)
            self._num_reflections = np.zeros(0, dtype=int)
            return
        # Run normal one-step raytracer
        rt = RayTracer(self._rayorigs, self._raydirs, self._meshlist)
        rt.run()

        n = self._raydirs.shape[0]
        intersection_mask = rt.get_intersection_mask()
        points_cartesic = np.zeros((n, 3))
        points_cartesic[intersection_mask] = rt.get_points_cartesic()
        points_barycentric = np.zeros((n, 3))
        points_barycentric[intersection_mask] = rt.get_points_barycentric()
        triangle_indices = -1 * np.ones(n, dtype=int)
        triangle_indices[intersection_mask] = rt.get_triangle_indices()
        mesh_indices = -1 * np.ones(n, dtype=int)
        mesh_indices[intersection_mask] = rt.get_mesh_indices()
        scale = np.zeros(n)
        scale[intersection_mask] = rt.get_scale()

        raydirs = self._raydirs
        num_reflections = np.zeros(n, dtype=int)

        total_num_reflections = 0
        while True:
            # Get rays whose reflections still need to be traced
            mirror_mask = np.logical_and(intersection_mask, self._mirrors[mesh_indices])
            # If there are none, we are done
            if not np.any(mirror_mask):
                break

            # Handle max number of reflections
            total_num_reflections += 1
            if total_num_reflections > self._max_num_reflections:
                # Mark rays that still need to be traced as misses
                intersection_mask[mirror_mask] = False
                break
            num_reflections[mirror_mask] += 1

            # Get normal vectors: TODO: Interpolate vertex normals!?
            midx = mesh_indices[mirror_mask]
            tidx = triangle_indices[mirror_mask]
            mirror_normals = [ np.asarray(self._meshlist[mi].triangle_normals[ti]) \
                for mi, ti in zip(midx, tidx) ]
            mirror_normals = np.vstack(mirror_normals)
            # Calculate ray dirs of reflected rays
            raydirs[mirror_mask] = RayTracerMirrors.__mirror_vector( \
                raydirs[mirror_mask], mirror_normals)
            # Get ray origins of reflected rays
            rayorigs = points_cartesic[mirror_mask]
            # TODO: document!
            myeps = 1e-3
            rayorigs = rayorigs + myeps * raydirs[mirror_mask]

            rt = RayTracer(rayorigs, raydirs[mirror_mask], self._meshlist)
            rt.run()

            intersection_mask[mirror_mask] = rt.get_intersection_mask()
            mirror_mask[mirror_mask] = rt.get_intersection_mask()

            points_cartesic[mirror_mask] = rt.get_points_cartesic()
            points_barycentric[mirror_mask] = rt.get_points_barycentric()
            triangle_indices[mirror_mask] = rt.get_triangle_indices()
            mesh_indices[mirror_mask] = rt.get_mesh_indices()
            scale[mirror_mask] += rt.get_scale() + myeps

        # Assemble final result
        self._intersection_mask = intersection_mask
        self._points_cartesic = points_cartesic[intersection_mask]
        self._points_barycentric = points_barycentric[intersection_mask]
        self._triangle_indices = triangle_indices[intersection_mask]
        self._mesh_indices = mesh_indices[intersection_mask]
        self._scale = scale[intersection_mask]
        self._num_reflections = num_reflections[intersection_mask]

