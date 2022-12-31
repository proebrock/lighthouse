import numpy as np

from camsimlib.shader import Shader
from camsimlib.projective_geometry import ProjectiveGeometry
from camsimlib.mesh_tools import get_points_normals_vertices



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
        # Temporary (?) fix of the incorrect determination of shadow points
        # due to P already lying inside the mesh and the raytracer
        # producing results with scale very close to zero
        triangle_idx = rt_result.triangle_indices
        triangle_normals = mesh.triangle_normals[triangle_idx]
        correction = 1e-3 * triangle_normals

        illu_mask = self._get_illuminated_mask_point_light(
            rt_result.points_cartesic + correction, mesh,
            self.get_pose().get_translation())

        # Project interconnection points of camera rays and to mesh to
        # chip of the projector in order to reconstruct colors for these points
        projector_colors, on_chip_mask = self._get_projector_colors( \
            rt_result.points_cartesic[illu_mask])
        # Update illumination mask of actually illuminated points
        illu_mask[illu_mask] = on_chip_mask

        points, normals, colors = get_points_normals_vertices( \
            mesh, rt_result, illu_mask)

        intensities = self._get_vertex_intensities_point_light(points,
            normals, self.get_pose().get_translation())

        projector_colors = projector_colors * intensities[:, np.newaxis]
        #object_colors = colors * intensities[:, np.newaxis]

        C = np.zeros_like(rt_result.points_cartesic)
        # TODO: we use just the projector_colors here; missing here is a model to
        # combine the color of the light (projector_colors) with the color of the
        # object at that position (object_colors)
        C[illu_mask] = projector_colors
        return C
