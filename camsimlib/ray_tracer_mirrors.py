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



    @staticmethod
    def __mirror_vector(vecs, normals):
        assert vecs.shape == normals.shape
        vn = np.sum(vecs * normals, axis=1)
        return vecs - 2 * vn[:, np.newaxis] * normals



    def run(self):
        """ Run ray tracing
        """
        # Run normal one-step raytracer
        rt = RayTracer(self._rayorigs, self._raydirs, self._meshlist)
        rt.run()
        # Copy results
        self._intersection_mask = rt.get_intersection_mask()
        self._points_cartesic = rt.get_points_cartesic()
        self._points_barycentric = rt.get_points_barycentric()
        self._triangle_indices = rt.get_triangle_indices()
        self._mesh_indices = rt.get_mesh_indices()
        self._scale = rt.get_scale()

        #while True:
        print()
        print(rt.get_intersection_mask())

        mirror_mask = self._mirrors[rt.get_mesh_indices()]
        print('mirror_mask', mirror_mask)
        if not np.any(mirror_mask):
            # If no ray hit a mirror we are done here
            return

        # Get normal vectors: TODO: Interpolate vertex normals!?
        midx = self._mesh_indices[mirror_mask]
        tidx = self._triangle_indices[mirror_mask]
        mirror_normals = [ np.asarray(self._meshlist[mi].triangle_normals[ti]) \
            for mi, ti in zip(midx, tidx) ]
        mirror_normals = np.vstack(mirror_normals)
        # Calculate ray dirs of reflected rays
        raydirs = self._raydirs[self._intersection_mask][mirror_mask]
        raydirs = RayTracerMirrors.__mirror_vector( \
            raydirs, mirror_normals)
        # Get ray origins of reflected rays
        rayorigs = self._points_cartesic[mirror_mask]
        # TODO
        myeps = 1e-3
        rayorigs = rayorigs + myeps * raydirs
        #with np.printoptions(precision=2, suppress=True):
        #    print('raydirs', raydirs)
        #    print('rayorigs', rayorigs)

        rt = RayTracer(rayorigs, raydirs, self._meshlist)
        rt.run()
        mirror_mask[mirror_mask] = rt.get_intersection_mask()
        self._intersection_mask[self._intersection_mask] = mirror_mask
        self._points_cartesic[mirror_mask] = rt.get_points_cartesic()
        self._points_barycentric[mirror_mask] = rt.get_points_barycentric()
        self._triangle_indices[mirror_mask] = rt.get_triangle_indices()
        self._mesh_indices[mirror_mask] = rt.get_mesh_indices()
        self._scale[mirror_mask] += rt.get_scale()

