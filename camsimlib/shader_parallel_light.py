import numpy as np

from camsimlib.ray_tracer_embree import RayTracer



class ShaderParallelLight:

    def __init__(self, light_direction):
        self._light_direction = np.asarray(light_direction)
        if self._light_direction.ndim != 1 or self._light_direction.size != 3:
            raise Exception(f'Invalid light direction {light_direction}')



    def __str__(self):
        return f'ShaderParallelLight(dir={self._light_direction})'



    def _get_shadow_points(self, P, mesh):
        # Vector from intersection point towards light source (no point)
        lightvecs = -self._light_direction
        lightvecs = lightvecs / np.linalg.norm(lightvecs)
        light_rt = RayTracer(P, lightvecs, mesh.vertices, mesh.triangles)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shade_points = light_rt.get_intersection_mask()
        shade_points[shade_points] = light_rt.get_scale() > 0.01 # TODO: some intersections pretty close to zero!
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

        # lightvecs are unit vectors from vertex toward the light
        lightvecs = -self._light_direction
        lightvecs = lightvecs / np.linalg.norm(lightvecs)
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
        shade_points = self._get_shadow_points(P, mesh)
        # Points in the shade only have 10% of the originally calculated brightness
        # TODO: More physically correct model? Make this configurable?
        # TODO: attenuation = 1.0 / (1.0 + k * distanceToLight**2) ?
        C[shade_points] *= 0.1
        return C
