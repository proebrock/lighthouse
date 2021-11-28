import numpy as np

from camsimlib.ray_tracer import RayTracer



class Shader:

    def __init__(self, ray_tracer, mesh, lighting_mode, light_vector):
        # Extract ray tracer results
        self.P = ray_tracer.get_points_cartesic() # shape (n, 3)
        self.Pbary = ray_tracer.get_points_barycentric() # shape (n, 3)
        self.triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )

        # Extract vertices and vertex normals from mesh
        self.mesh = mesh
        self.triangles = np.asarray(self.mesh.triangles)[self.triangle_idx, :] # shape (n, 3)
        self.vertices = np.asarray(self.mesh.vertices)[self.triangles] # shape (n, 3, 3)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)[self.triangles] # shape (n, 3, 3)

        if lighting_mode not in ('cam', 'point', 'parallel'):
            raise ValueError(f'Unknown lighting mode "{lighting_mode}')
        self.lighting_mode = lighting_mode
        self.light_vector = np.asarray(light_vector)
        if light_vector.ndim != 1 or light_vector.size != 3:
            raise Exception(f'Invalid light vector {light_vector}')



    def get_shadow_points(self):
        if self.lighting_mode == 'cam':
            # When lighting mode is 'cam', camera and point light source are at the same point;
            # so if a point is visible by the camera, it is lighted by the light source; there
            # is no way a point could lie in the shadow
            return np.zeros((self.P.shape[0], ), dtype=bool)
        elif self.lighting_mode == 'point':
            # Vector from intersection point camera-mesh toward point light source
            lightvecs = -self.P + self.light_vector
            light_rt = RayTracer(self.P, lightvecs, self.mesh.vertices, self.mesh.triangles)
            light_rt.run()
            # When there is some part of the mesh between the intersection point camera-mesh
            # and the light source, the point lies in shade
            shade_points = light_rt.get_intersection_mask()
            # When scale is in [0..1], the mesh is between intersection point and light source;
            # if scale is >1, the mesh is behind the light source, so there is no intersection!
            shade_points[shade_points] = light_rt.get_scale() < 1.0
            return shade_points
        elif self.lighting_mode == 'parallel':
            # Vector from intersection point towards light source (no point)
            lightvecs = -self.light_vector
            lightvecs = lightvecs / np.linalg.norm(lightvecs)
            light_rt = RayTracer(self.P, lightvecs, self.mesh.vertices, self.mesh.triangles)
            light_rt.run()
            # When there is some part of the mesh between the intersection point camera-mesh
            # and the light source, the point lies in shade
            shade_points = light_rt.get_intersection_mask()
            shade_points[shade_points] = light_rt.get_scale() > 0.01 # TODO: some intersections pretty close to zero!
            return shade_points
        else:
            raise Exception(f'Unknown lighting mode {self.lighting_mode}')



    def run(self):
        if self.lighting_mode == 'cam' or self.lighting_mode == 'point':
            # We assume that at position self.light_vector there is a point light source
            # lightvecs are unit vectors from vertex to light source
            lightvecs = -self.vertices + self.light_vector
            # Normalize light vectors
            lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        elif self.lighting_mode == 'parallel':
            lightvecs = -self.light_vector
            # Normalize light vectors
            lightvecs = lightvecs / np.linalg.norm(lightvecs)
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(self.vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
        # From intensity determine color shade
        if self.mesh.has_vertex_colors():
            vertex_colors = np.asarray(self.mesh.vertex_colors)[self.triangles]
        else:
            vertex_colors = np.ones((self.triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        C = np.einsum('ijk, ij->ik', vertex_color_shades, self.Pbary)
        shade_points = self.get_shadow_points()
        # Points in the shade only have 10% of the originally calculated brightness
        # TODO: More physically correct model? Make this configurable?
        C[shade_points] *= 0.1
        return C
