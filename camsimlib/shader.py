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

        self.lighting_mode = lighting_mode
        self.light_vector = np.asarray(light_vector)
        if light_vector.ndim != 1 or light_vector.size != 3:
            raise Exception(f'Invalid light vector {light_vector}')



    def get_points_reached_by_light_mask(self):
        print(f'Number of points: {self.P.shape[0]}')
        if self.lighting_mode == 'point':
            lightvecs = -self.P + self.light_vector
            lightvecs /= np.linalg.norm(lightvecs, axis=1)[:, np.newaxis]
            light_rt = RayTracer(self.P, lightvecs, self.mesh.vertices, self.mesh.triangles)
            light_rt.run()
            P = light_rt.get_points_cartesic()
            print(f'Number of points: {P.shape[0]}')



    def run(self):
        self.get_points_reached_by_light_mask()

        if self.lighting_mode == 'cam':
            # We assume that at position self.light_vector there is a point light source
            # lightvecs are unit vectors from vertex to light source
            lightvecs = -self.vertices + self.light_vector
            lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
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
            return np.einsum('ijk, ij->ik', vertex_color_shades, self.Pbary)
        else:
            raise Exception(f'Unknown lighting mode {self.lighting_mode}')
