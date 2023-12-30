import numpy as np

from camsimlib.shader import Shader
from camsimlib.projective_geometry import ProjectiveGeometry
from camsimlib.ray_tracer_result import get_points_normals_colors



class ShaderProjector(Shader, ProjectiveGeometry):

    def __init__(self, image=np.zeros((30, 40, 3), dtype=np.uint8),
                max_intensity=1.0, focal_length=100, principal_point=None,
                distortion=None, pose=None):
        self.set_image(image)
        Shader.__init__(self, max_intensity)
        ProjectiveGeometry.__init__(self, focal_length,
            principal_point, distortion, pose)



    def __str__(self):
        return (super().__str__() +
                ', ShaderProjector(' +
                f'chip_size={self.get_chip_size()}, ' +
                ')'
                )



    def get_chip_size(self):
        chip_size = np.array((self._image.shape[1],
            self._image.shape[0]), dtype=int)
        return chip_size



    def get_image(self):
        return self._image



    def set_image(self, image):
        assert image.ndim == 3 # RBG image
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
        self._image = image.astype(float) / 255.0



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        super(ShaderProjector, self).dict_save(param_dict)
        param_dict['image_shape'] = self._image.shape[0:2]



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        super(ShaderProjector, self).dict_load(param_dict)
        shape = param_dict['image_shape']
        self._image = np.zeros((*shape, 3))



    def _get_projector_colors(self, P):
        # Project points to projector chip
        p = self.scene_to_chip(P)
        # Round coordinates to nearest pixel
        indices = np.round(p[:, 0:2]).astype(int)
        on_chip_mask = self.points_on_chip_mask(indices)
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

        points, normals, object_colors = get_points_normals_colors( \
            mesh, rt_result, illu_mask)

        intensities = self._get_vertex_intensities_point_light(points,
            normals, self.get_pose().get_translation())

        # Object color and the light color are combined with so-called
        # "multiplicative blending" to get the final color sensed by the camera
        # https://en.wikipedia.org/wiki/Blend_modes#Multiply
        colors = (projector_colors * object_colors) * intensities[:, np.newaxis]

        C = np.zeros_like(rt_result.points_cartesic)
        C[illu_mask] = colors
        return C
