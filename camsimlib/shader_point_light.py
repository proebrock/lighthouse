import numpy as np

from camsimlib.ray_tracer_embree import RayTracer



class ShaderPointLight:

    def __init__(self, light_position):
        self._light_position = np.asarray(light_position)
        if self._light_position.ndim != 1 or self._light_position.size != 3:
            raise Exception(f'Invalid light position {light_position}')



    def __str__(self):
        return f'ShaderPointLight(light_position={self._light_position})'



    def _get_shadow_points(self, P, mesh):
        # Vector from intersection point camera-mesh toward point light source
        lightvecs = -P + self._light_position
        light_rt = RayTracer(P, lightvecs, mesh.vertices, mesh.triangles)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shade_points = light_rt.get_intersection_mask()
        # When scale is in [0..1], the mesh is between intersection point and light source;
        # if scale is >1, the mesh is behind the light source, so there is no intersection!
        shade_points[shade_points] = np.logical_and( \
            light_rt.get_scale() > 0.01, # TODO: some intersections pretty close to zero!
            light_rt.get_scale() < 1.0)
        return shade_points



    def run(self, cam, ray_tracer, mesh):
        # Extract ray tracer results
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        Pbary = ray_tracer.get_points_barycentric() # shape (n, 3)
        triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(mesh.vertex_normals)[triangles] # shape (n, 3, 3)

        # lightvecs are unit vectors from vertex to light source
        lightvecs = -vertices + self._light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
        # From intensity determine color shade
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)[triangles]
        else:
            vertex_colors = np.ones((triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        C = np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)
        # In case point light source is located at camera position,
        # there are no shade points
        if not np.allclose(self._light_position, cam.get_pose().get_translation()):
            shadow_points = self._get_shadow_points(P, mesh)
            # Points in the shade only have 10% of the originally calculated brightness
            # TODO: More physically correct model? Make this configurable?
            # TODO: attenuation = 1.0 / (1.0 + k * distanceToLight**2) ?
            C[shadow_points] *= 0.1
        return C

