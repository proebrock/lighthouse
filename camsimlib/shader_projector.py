import numpy as np

from camsimlib.shader import Shader
from camsimlib.projective_geometry import ProjectiveGeometry



class ShaderProjector(Shader, ProjectiveGeometry):

    def __init__(self, image, max_intensity=1.0, focal_length=100,
                principal_point=None, distortion=None, pose=None):
        self._image = image
        Shader.__init__(self, max_intensity)
        ProjectiveGeometry.__init__(self, focal_length,
            principal_point, distortion, pose)



    def __str__(self):
        return f'ShaderPointLight(super(ShaderProjector, self).__str__())'



    def get_chip_size(self):
        chip_size = np.array((self._image.shape[1],
            self._image.shape[0]), dtype=int)
        return chip_size



    def get_image(self):
        return self._image



    def set_image(self, image):
        self._image = image



    def _get_projector_colors(self, P):
        # Project points to projector chip
        p = self.scene_to_chip(P)
        # Round coordinates to nearest pixel
        indices = np.round(p[:, 0:2]).astype(int)
        on_chip_mask = np.logical_and.reduce((
            indices[:, 0] >= 0,
            indices[:, 0] < self.get_chip_size()[0],
            indices[:, 1] >= 0,
            indices[:, 1] < self.get_chip_size()[1],
        ))
        indices = indices[on_chip_mask, :]
        point_colors = self._image[indices[:, 1], indices[:, 0], :]
        return point_colors, on_chip_mask



    def run(self, cam, rt_result, mesh):
        # Extract ray tracer results
        P = rt_result.points_cartesic # shape (n, 3)

        # Prepare shader result
        C = np.zeros_like(P)

        # In case the light source is located at the same position as the camera,
        # all points are illuminated by the light source, there are not shadow points
        if np.allclose(self.get_pose().get_translation(), \
            cam.get_pose().get_translation()):
            illu_mask = np.ones(P.shape[0], dtype=bool)
        else:
            # Temporary (?) fix of the incorrect determination of shadow points
            # due to P already lying inside the mesh and the raytracer
            # producing results with scale very close to zero
            triangle_idx = rt_result.triangle_indices
            triangle_normals = mesh.triangle_normals[triangle_idx]
            correction = 1e-3 * triangle_normals

            illu_mask = self._get_illuminated_mask_point_light(P + correction, mesh,
            self.get_pose().get_translation())
        #print(f'Number of points not in shadow {np.sum(illu_mask)}')

        # Project interconnection points of camera rays and to mesh to
        # chip of the projector in order to reconstruct colors for these points
        projector_colors, on_chip_mask = self._get_projector_colors(P[illu_mask])
        # Update illumination mask of actually illuminated points
        illu_mask[illu_mask] = on_chip_mask
        #print(f'Number of points on projector chip {np.sum(illu_mask)}')

        # Extract ray tracer results and mesh elements
        P = P[illu_mask, :] # shape (n, 3)
        Pbary = rt_result.points_barycentric[illu_mask, :] # shape (n, 3)
        triangle_idx = rt_result.triangle_indices[illu_mask] # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = mesh.triangles[triangle_idx, :] # shape (n, 3)
        vertices = mesh.vertices[triangles] # shape (n, 3, 3)
        vertex_normals = mesh.vertex_normals[triangles] # shape (n, 3, 3)

        vertex_intensities = self._get_vertex_intensities_point_light(vertices,
            vertex_normals, self.get_pose().get_translation())  # shape: (n, 3)

        point_intensities = np.sum(vertex_intensities * Pbary, axis=1)
        projector_colors = projector_colors * point_intensities[:, np.newaxis]

        # From vertex intensities determine object colors
        if mesh.has_vertex_colors():
            vertex_colors = mesh.vertex_colors[triangles]
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
