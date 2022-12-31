import numpy as np
import copy

from camsimlib.ray_tracer import RayTracer as RayTracerBaseClass
from camsimlib.ray_tracer_embree import RayTracerEmbree as RayTracer

from camsimlib.ray_tracer_result import RayTracerResult
from camsimlib.rays import Rays
from camsimlib.mesh_tools import get_triangle_normals, get_interpolated_vertex_normals



class RayTracerMirrors(RayTracerBaseClass):

    def __init__(self, rays, meshes):
        super(RayTracerMirrors, self).__init__(rays, meshes)
        # Maximum number  of reflections before ray tracing is aborted
        self._max_num_reflections = 10



    @staticmethod
    def __mirror_vector(vecs, normals):
        assert vecs.shape == normals.shape
        vn = np.sum(vecs * normals, axis=1)
        return vecs - 2 * vn[:, np.newaxis] * normals



    def run(self):
        """ Run ray tracing
        """
        # Reset result and handle trivial case
        self.r.clear()
        if self._mesh.num_triangles() == 0:
            self.r.intersection_mask = np.zeros(len(self._rays), dtype=bool)
            return
        # Initially run normal one-step raytracer
        rt = RayTracer(self._rays, self._mesh)
        rt.run()

        self.r = copy.copy(rt.r)
        self.r.expand()
        rays = copy.copy(self._rays)

        total_num_reflections = 0
        while True:
            # Get rays whose reflections still need to be traced
            mirror_mask = np.logical_and(self.r.intersection_mask, \
                self._mesh.triangle_is_mirror[self.r.triangle_indices])
            # If there are none, we are done
            if not np.any(mirror_mask):
                break

            # Handle max number of reflections and reflection counters
            total_num_reflections += 1
            if total_num_reflections > self._max_num_reflections:
                # Mark rays that still need to be traced as misses
                self.r.intersection_mask[mirror_mask] = False
                break
            self.r.num_reflections[mirror_mask] += 1

            #mirror_normals = get_triangle_normals(self._mesh, self.r, mirror_mask)
            mirror_normals = get_interpolated_vertex_normals(self._mesh, self.r, mirror_mask)

            # Calculate ray dirs of reflected rays
            rays.dirs[mirror_mask] = RayTracerMirrors.__mirror_vector( \
                rays.dirs[mirror_mask], mirror_normals)
            # Get ray origins of reflected rays
            rays.origs = self.r.points_cartesic
            # Same problem as with continuing raytracing in the shader to determine
            # shadow points: if we continue tracing a ray with a start point that lies
            # exactly on the mesh, the next ray tracing step may just produce the same
            # result again with a scale of zero...
            # To fight this problem we move a little bit away from the previous point
            # in the direction of the new ray. This should ensure that we do not end
            # up with the same raytracing result as in the last step.
            # Problem is how far to move along this ray? We guess we deal with stuctures
            # in the size of millimeters, so this epsilon may be a good choice (?).
            myeps = 1e-3
            rays.origs[mirror_mask] = rays.origs[mirror_mask] + myeps * rays.dirs[mirror_mask]

            rt = RayTracer(rays.filter(mirror_mask), self._mesh)
            rt.run()

            self.r.intersection_mask[mirror_mask] = rt.r.intersection_mask
            mirror_mask[mirror_mask] = rt.r.intersection_mask
            self.r.points_cartesic[mirror_mask] = rt.r.points_cartesic
            self.r.points_barycentric[mirror_mask] = rt.r.points_barycentric
            self.r.triangle_indices[mirror_mask] = rt.r.triangle_indices
            self.r.scale[mirror_mask] += rt.r.scale + myeps

        # Reduce result to real intersections
        self.r.reduce_to_valid()
