import numpy as np



from camsimlib.ray_tracer import RayTracer



class RayTracerMirrors(RayTracer):

    def __init__(self, rayorigs, raydirs, solid_meshes, mirror_meshes=[]):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param solid_meshes: List of solid meshes
        :param mirror_meshes: List of mirror meshes
        """
        super(RayTracerMirrors, self).__init__(rayorigs, raydirs)
        self._solid_meshes = solid_meshes
        self._mirror_meshes = mirror_meshes



    def run(self):
        """ Run ray tracing (parallel processing)
        """
        pass
