import numpy as np

from camsimlib.projective_geometry import ProjectiveGeometry
from camsimlib.ray_tracer_embree import RayTracer



class ShaderProjector(ProjectiveGeometry):

    def __init__(self, image, focal_length=100, principal_point=None,
                 distortion=None, pose=None):
        chip_size = (image.shape[1], image.shape[0])
        super(ShaderProjector, self).__init__(chip_size, focal_length,
            principal_point, distortion, pose)
        self._image = image



    def get_image(self):
        return self._image



    def set_image(self, image):
        chip_size = (image.shape[1], image.shape[0])
        super(ShaderProjector, self).set_chip_size(chip_size)
        self._image = image



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
            light_rt.get_scale() > 0.01, # TODO: some intersections pretty close to zero!
            light_rt.get_scale() < 1.0)
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



    def _get_projector_colors(self, P):
        # Project points to projector chip
        p = self.scene_to_chip(P)
        # Round coordinates to nearest pixel
        indices = np.round(p[:, 0:2]).astype(int)
        on_chip_mask = np.logical_and.reduce((
            indices[:, 0] >= 0,
            indices[:, 0] < self._chip_size[0],
            indices[:, 1] >= 0,
            indices[:, 1] < self._chip_size[1],
        ))
        indices = indices[on_chip_mask, :]
        point_colors = self._image[indices[:, 1], indices[:, 0], :]
        return point_colors, on_chip_mask



    def run(self, cam, ray_tracer, mesh):
        # Extract ray tracer results
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        print(f'Number of camera rays {ray_tracer.get_intersection_mask().size}')
        print(f'Number of intersections with mesh {P.shape[0]}')

        # Prepare shader result
        C = np.zeros_like(P)

        # In case the light source is located at the same position as the camera,
        # all points are illuminated by the light source, there are not shadow points
        if np.allclose(self.get_pose().get_translation(), \
            cam.get_pose().get_translation()):
            illu_mask = np.ones(P.shape[0], dtype=bool)
        else:
            illu_mask = self._get_illuminated_mask_point_light(P, mesh,
            self.get_pose().get_translation())
        print(f'Number of points not in shadow {np.sum(illu_mask)}')

        # Project interconnection points of camera rays and to mesh to
        # chip of the projector in order to reconstruct colors for these points
        projector_colors, on_chip_mask = self._get_projector_colors(P[illu_mask])
        # Update illumination mask of actually illuminated points
        illu_mask[illu_mask] = on_chip_mask
        print(f'Number of points on projector chip {np.sum(illu_mask)}')

        # Extract ray tracer results and mesh elements
        P = P[illu_mask, :] # shape (n, 3)
        Pbary = ray_tracer.get_points_barycentric()[illu_mask, :] # shape (n, 3)
        triangle_idx = ray_tracer.get_triangle_indices()[illu_mask] # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(mesh.vertex_normals)[triangles] # shape (n, 3, 3)

        vertex_intensities = self._get_vertex_intensities(vertices, vertex_normals,
            self.get_pose().get_translation())  # shape: (n, 3)

        point_intensities = np.sum(vertex_intensities * Pbary, axis=1)
        projector_colors = projector_colors * point_intensities[:, np.newaxis]

        # From vertex intensities determine object colors
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)[triangles]
        else:
            vertex_colors = np.ones((triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        object_colors = np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)

        # TODO: we use just the projector_colors here; missing here is a model to
        # combine the color of the light (projector_colors) with the color of the
        # object at that position (object_colors)
        C[illu_mask] = projector_colors
        return C
