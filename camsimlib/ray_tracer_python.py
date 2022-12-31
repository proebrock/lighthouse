import numpy as np
import multiprocessing



from camsimlib.ray_tracer import RayTracer



class RayTracerPython(RayTracer):

    def __init__(self, rays, meshes):
        super(RayTracerPython, self).__init__(rays, meshes)
        self._triangle_vertices = self._mesh.vertices[ \
            self._mesh.triangles]



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



    def ray_mesh_intersect(self, rayindex):
        """ Intersection of a single ray with the triangles of the mesh
        Tests intersection of ray with all triangles and returns the one with lowest Z coordinate
        Based on Möller–Trumbore intersection algorithm (see https://scratchapixel.com)
        """
        # Switch off warnings about divide by zero and invalid float op:
        # We do some batch-computations and check the validity of the results later
        with np.errstate(divide='ignore', invalid='ignore'):
            num_tri = self._triangle_vertices.shape[0]
            rays = np.tile(self._rays.dirs[rayindex], num_tri).reshape((num_tri, 3))
            # Do all calculation no matter if invalid values occur during calculation
            v0 = self._triangle_vertices[:, 0, :]
            v0v1 = self._triangle_vertices[:, 1, :] - v0
            v0v2 = self._triangle_vertices[:, 2, :] - v0
            pvec = RayTracerPython.__cross(rays, v0v2)
            det = RayTracerPython.__multsum(v0v1, pvec)
            inv_det = 1.0 / det
            tvec = self._rays.origs[rayindex] - v0
            u = inv_det * RayTracerPython.__multsum(tvec, pvec)
            qvec = RayTracerPython.__cross(tvec, v0v1)
            v = inv_det * RayTracerPython.__multsum(rays, qvec)
            t = inv_det * RayTracerPython.__multsum(v0v2, qvec)
            # Check all results for validity
            invalid = np.logical_or.reduce((
                np.isclose(det, 0.0),
                u < 0.0,
                u > 1.0,
                v < 0.0,
                (u + v) > 1.0,
                np.isclose(t, 0.0),
                t < 0.0,
            ))
            valid_idx = np.where(~invalid)[0]
            if valid_idx.size == 0:
                # No intersection of ray with any triangle in mesh
                return np.NaN * np.zeros(3+3+1+1)
            else:
                # triangle_index is the index of the triangle intersection point with
                # the lowest t, which means it is the intersection point closest to the camera
                triangle_index = valid_idx[t[valid_idx].argmin()]
                # Prepare result
                result = np.zeros(3+3+1+1)
                # Cartesic intersection point
                result[0:3] = self._rays.origs[rayindex] + self._rays.dirs[rayindex] * t[triangle_index]
                # Barycentric intersection point
                result[3] = 1.0 - u[triangle_index] - v[triangle_index]
                result[4] = u[triangle_index]
                result[5] = v[triangle_index]
                # Triangle index
                result[6] = triangle_index
                # Scale
                result[7] = t[triangle_index]
                return result



    def run_serial(self):
        """ Run ray tracing (serial processing)
        Is useful for profiling the software without the problem of having multiple processes
        """
        # Reset result and handle trivial case
        self.r.clear()
        if self._mesh.num_triangles() == 0:
            self.r.intersection_mask = np.zeros(len(self._rays), dtype=bool)
            return
        # Prepare results
        result = np.zeros((len(self._rays), 3+3+1+1))
        # Switch off warnings about divide by zero and invalid float op
        with np.errstate(divide='ignore', invalid='ignore'):
            for i in range(len(self._rays)):
                result[i, :] = self.ray_mesh_intersect(i)
        # Extract results and reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 6])
        self.r.intersection_mask = valid
        self.r.initial_points_cartesic = result[valid, 0:3]
        self.r.points_cartesic = result[valid, 0:3]
        self.r.points_barycentric = result[valid, 3:6]
        self.r.triangle_indices = result[valid, 6].astype(int)
        self.r.scale = result[valid, 7]
        self.r.num_reflections = np.zeros_like(self.r.triangle_indices, dtype=int)



    def run(self):
        """ Run ray tracing (parallel processing)
        """
        # Reset result and handle trivial case
        self.r.clear()
        if self._mesh.num_triangles() == 0:
            self.r.intersection_mask = np.zeros(len(self._rays), dtype=bool)
            return
        # Run
        pool = multiprocessing.Pool()
        result_list = pool.map(self.ray_mesh_intersect, range(len(self._rays)))
        result = np.asarray(result_list)
        # Extract results and reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(result[:, 6])
        self.r.intersection_mask = valid
        self.r.initial_points_cartesic = result[valid, 0:3]
        self.r.points_cartesic = result[valid, 0:3]
        self.r.points_barycentric = result[valid, 3:6]
        self.r.triangle_indices = result[valid, 6].astype(int)
        self.r.scale = result[valid, 7]
        self.r.num_reflections = np.zeros_like(self.r.triangle_indices, dtype=int)
