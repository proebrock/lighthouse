import numpy as np

from camsimlib.ray_tracer import RayTracer



class Shader:

    def __init__(self, mesh, lighting_mode, light_vector):
        self.mesh = mesh
        self.lighting_mode = lighting_mode
        self.light_vector = np.asarray(light_vector)
        if light_vector.ndim != 1 or light_vector.size != 3:
            raise Exception(f'Invalid light vector {light_vector}')



    def run(self, ray_tracer):
        # Extract vertices and vertex normals from mesh
        triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )
        triangles = np.asarray(self.mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(self.mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(self.mesh.vertex_normals)[triangles] # shape (n, 3, 3)
        if self.lighting_mode == 'cam':
            # We assume that at position self.light_vector there is a point light source
            # lightvecs are unit vectors from vertex to light source
            lightvecs = -vertices + self.light_vector
            lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
            # Dot product of vertex_normals and lightvecs; if angle between
            # those is 0°, the intensity is 1; the intensity decreases up
            # to an angle of 90° where it is 0
            vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
            vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
            # From intensity determine color shade
            if self.mesh.has_vertex_colors():
                vertex_colors = np.asarray(self.mesh.vertex_colors)[triangles]
            else:
                vertex_colors = np.ones((triangles.shape[0], 3, 3))
            vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
            # Interpolate to get color of intersection point
            Pbary = ray_tracer.get_points_barycentric()
            return np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)
        else:
            raise Exception(f'Unknown lighting mode {self.lighting_mode}')
