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



#    def set_chip_size(self, chip_size):
#        raise Exception('Chip size of ShaderProjector is determined by image provided and cannot be set separately.')



    def scale_resolution(self, factor=1.0):
        super(ShaderProjector, self).scale_resolution(factor)
        # TODO: scale self._image as well!!!???!!!



    def _get_illuminated_mask(self, P, mesh):
        # Vector from intersection point camera-mesh toward point light source
        lightvecs = -P + self.get_pose().get_translation()
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



    def _get_projector_colors(self, P):
        pass



    def run(self, cam, ray_tracer, mesh):
        # Extract ray tracer results
        P = ray_tracer.get_points_cartesic() # shape (n, 3)
        Pbary = ray_tracer.get_points_barycentric() # shape (n, 3)
        triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(mesh.vertex_normals)[triangles] # shape (n, 3, 3)
        print(f'Number of camera rays {ray_tracer.get_intersection_mask().size}')
        print(f'Number of intersections with mesh {P.shape[0]}')

        # In case the light source is located at the same position as the camera,
        # all points are illuminated by the light source, there are not shadow points
        if np.allclose(self.get_pose().get_translation(), \
            cam.get_pose().get_translation()):
            illu_mask = np.ones(P.shape[0], dtype=bool)
        else:
            illu_mask = self._get_illuminated_mask(P, mesh)
        print(f'Number of points not in shadow {np.sum(illu_mask)}')

        # Project points to camera chip
        p = self.scene_to_chip(P[illu_mask])
        # Get mask of points projected to chip of projector
        on_chip_mask = np.logical_and.reduce((
            p[:, 0] >= 0,
            p[:, 0] <= self._chip_size[0], # TODO <= or < ???
            p[:, 1] >= 0,
            p[:, 1] <= self._chip_size[1], # TODO <= or < ???
        ))
        print(f'Number of points on projector chip {np.sum(on_chip_mask)}')

        # TODO: Interpolate/sample colors in self._image to get point color

        # TODO: Reduce P, Pbary, ... in order to calculate intensity/vertex_colors




        C = np.ones_like(P)
        return C
