import numpy as np

from camsimlib.ray_tracer import RayTracer



class ShaderPointLight:

    def __init__(self, light_position):
        self.light_position = np.asarray(light_position)
        if self.light_position.ndim != 1 or self.light_position.size != 3:
            raise Exception(f'Invalid light position {light_position}')



    def __str__(self):
        return f'ShaderPointLight(pos={self.light_position})'



    def get_shadow_points(self):
        # Vector from intersection point camera-mesh toward point light source
        lightvecs = -self.P + self.light_position
        light_rt = RayTracer(self.P, lightvecs, self.mesh.vertices, self.mesh.triangles)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shade_points = light_rt.get_intersection_mask()
        # When scale is in [0..1], the mesh is between intersection point and light source;
        # if scale is >1, the mesh is behind the light source, so there is no intersection!
        shade_points[shade_points] = light_rt.get_scale() < 1.0
        return shade_points



    def run(self, cam, ray_tracer, mesh):
        # Extract ray tracer results
        self.P = ray_tracer.get_points_cartesic() # shape (n, 3)
        self.Pbary = ray_tracer.get_points_barycentric() # shape (n, 3)
        self.triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )
        # Extract vertices and vertex normals from mesh
        self.mesh = mesh
        self.triangles = np.asarray(self.mesh.triangles)[self.triangle_idx, :] # shape (n, 3)
        self.vertices = np.asarray(self.mesh.vertices)[self.triangles] # shape (n, 3, 3)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)[self.triangles] # shape (n, 3, 3)

        # TODO: cleanup

        # lightvecs are unit vectors from vertex to light source
        lightvecs = -self.vertices + self.light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(self.vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
        # From intensity determine color shade
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)[self.triangles]
        else:
            vertex_colors = np.ones((self.triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        C = np.einsum('ijk, ij->ik', vertex_color_shades, self.Pbary)
        # In case point light source is located at camera position,
        # there are no shade points
        if not np.allclose(self.light_position, cam.get_camera_pose().get_translation()):
            shade_points = self.get_shadow_points()
            # Points in the shade only have 10% of the originally calculated brightness
            # TODO: More physically correct model? Make this configurable?
            C[shade_points] *= 0.1
        return C

