import numpy as np

from camsimlib.ray_tracer_embree import RayTracer



class ShaderPointLight:

    def __init__(self, light_position):
        self._light_position = np.asarray(light_position)
        if self._light_position.ndim != 1 or self._light_position.size != 3:
            raise Exception(f'Invalid light position {light_position}')



    def __str__(self):
        return f'ShaderPointLight(light_position={self._light_position})'



    def _get_illuminated_mask_point_light(self, P, mesh, light_position):
        # Vector from intersection point camera-mesh toward point light source
        lightvecs = -P + light_position
        light_rt = RayTracer(P, lightvecs, mesh.vertices, mesh.triangles)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shadow_points = light_rt.get_intersection_mask()
        # When scale is in [0..1], the mesh is between intersection point and light source;
        # if scale is >1, the mesh is behind the light source, so there is no intersection!
        shadow_points[shadow_points] = np.logical_and( \
            light_rt.get_scale() >= 0.0,
            light_rt.get_scale() <= 1.0)
        return ~shadow_points



    def _get_vertex_intensities(self, vertices, vertex_normals, light_position):
        # lightvecs are unit vectors from vertex to light source
        lightvecs = -vertices + light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
        return vertex_intensities



    def run(self, cam, ray_tracer, mesh):
        # Extract ray tracer results
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        print(f'Number of camera rays {ray_tracer.get_intersection_mask().size}')
        print(f'Number of intersections with mesh {P.shape[0]}')

        # Prepare shader result
        C = np.zeros_like(P)

        # In case the light source is located at the same position as the camera,
        # all points are illuminated by the light source, there are not shadow points
        if np.allclose(self._light_position, \
            cam.get_pose().get_translation()):
            illu_mask = np.ones(P.shape[0], dtype=bool)
        else:
            # Temporary (?) fix of the incorrect determination of shadow points
            # due to P already lying inside the mesh and the raytracer
            # producing results with scale very close to zero
            triangle_idx = ray_tracer.get_triangle_indices()
            triangle_normals = np.asarray(mesh.triangle_normals)[triangle_idx]
            correction = 1e-3 * triangle_normals

            illu_mask = self._get_illuminated_mask_point_light(P + correction, mesh,
            self._light_position)
        print(f'Number of points not in shadow {np.sum(illu_mask)}')

        # Extract ray tracer results and mesh elements
        P = P[illu_mask, :] # shape (n, 3)
        Pbary = ray_tracer.get_points_barycentric()[illu_mask, :] # shape (n, 3)
        triangle_idx = ray_tracer.get_triangle_indices()[illu_mask] # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(mesh.vertex_normals)[triangles] # shape (n, 3, 3)

        vertex_intensities = self._get_vertex_intensities(vertices, vertex_normals,
            self._light_position)  # shape: (n, 3)

        # From vertex intensities determine object colors
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)[triangles]
        else:
            vertex_colors = np.ones((triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        object_colors = np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)

        C[illu_mask] = object_colors
        return C
