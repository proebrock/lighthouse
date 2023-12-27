from abc import ABC, abstractmethod
import numpy as np

from camsimlib.rays import Rays
from camsimlib.ray_tracer_embree import RayTracerEmbree as RayTracer



class Shader(ABC):

    def __init__(self, max_intensity=1.0):
        self._max_intensity = max_intensity



    def get_max_intensity(self):
        return self._max_intensity



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        super(Shader, self).dict_save(param_dict)
        param_dict['max_intensity'] = self._max_intensity



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        super(Shader, self).dict_load(param_dict)
        self._max_intensity = param_dict['max_intensity']



    def _get_illuminated_mask_point_light(self, P, mesh, light_position):
        # Special case: no points: return empty mask
        if P.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        # Vector from intersection point camera-mesh toward point light source
        rays = Rays(P, -P + light_position)
        light_rt = RayTracer(rays, mesh)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shadow_points = light_rt.r.intersection_mask
        # When scale is in [0..1], the mesh is between intersection point and light source;
        # if scale is >1, the mesh is behind the light source, so there is no intersection!
        shadow_points[shadow_points] = np.logical_and( \
            light_rt.r.scale >= 0.0,
            light_rt.r.scale <= 1.0)
        return ~shadow_points



    def _get_illuminated_mask_parallel_light(self, P, mesh, light_direction):
        # Special case: no points: return empty mask
        if P.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        # Vector from intersection point towards light source (no point)
        rays = Rays(P, -light_direction)
        rays.normalize()
        light_rt = RayTracer(rays, mesh)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shadow_points = light_rt.r.intersection_mask
        return ~shadow_points



    def _get_vertex_intensities_point_light(self, vertices, vertex_normals, light_position):
        # lightvecs are unit vectors from vertex to light source
        lightvecs = -vertices + light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=1)[:, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=1)
        vertex_intensities = np.clip(vertex_intensities, 0.0, self._max_intensity)
        return vertex_intensities



    def _get_vertex_intensities_parallel_light(self, vertex_normals, light_direction):
        # lightvecs are unit vectors from vertex toward the light
        lightvecs = -light_direction
        lightvecs = lightvecs / np.linalg.norm(lightvecs)
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=1)
        vertex_intensities = np.clip(vertex_intensities, 0.0, self._max_intensity)
        return vertex_intensities



    @abstractmethod
    def run(self, cam, rt_result, mesh):
        pass
