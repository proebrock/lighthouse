from abc import ABC, abstractmethod
import numpy as np


from camsimlib.ray_tracer_embree import RayTracerEmbree as RayTracer



class Shader(ABC):

    def __init__(self, max_intensity=1.0):
        self._max_intensity = max_intensity



    def get_max_intensity(self):
        return self._max_intensity



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



    def _get_illuminated_mask_parallel_light(self, P, mesh, light_direction):
        # Vector from intersection point towards light source (no point)
        lightvecs = -light_direction
        lightvecs = lightvecs / np.linalg.norm(lightvecs)
        light_rt = RayTracer(P, lightvecs, mesh.vertices, mesh.triangles)
        light_rt.run()
        # When there is some part of the mesh between the intersection point camera-mesh
        # and the light source, the point lies in shade
        shadow_points = light_rt.get_intersection_mask()
        return ~shadow_points



    def _get_vertex_intensities_point_light(self, vertices, vertex_normals, light_position):
        # lightvecs are unit vectors from vertex to light source
        lightvecs = -vertices + light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, self._max_intensity)
        return vertex_intensities



    def _get_vertex_intensities_parallel_light(self, vertex_normals, light_direction):
        # lightvecs are unit vectors from vertex toward the light
        lightvecs = -light_direction
        lightvecs = lightvecs / np.linalg.norm(lightvecs)
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, self._max_intensity)
        return vertex_intensities



    @abstractmethod
    def run(self, cam, ray_tracer, mesh):
        pass
