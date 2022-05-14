# -*- coding: utf-8 -*-
""" Simulation of depth and/or RGB cameras
"""

import numpy as np
import open3d as o3d

from camsimlib.projective_geometry import ProjectiveGeometry
from camsimlib.ray_tracer_embree import RayTracer
from camsimlib.shader_point_light import ShaderPointLight



class CameraModel(ProjectiveGeometry):
    """ Class for simulating a depth and/or RGB camera

    Camera coordinate system with Z-Axis pointing into direction of view

    Z               X - Axis
                    self._chip_size[0] = width
      X--------->   depth_image second dimension
      |
      |
      |
      |
      V

    Y - Axis
    self._chip_size[1] = height
    depth_image first dimension

    """
    def __init__(self, chip_size=(40, 30), focal_length=100, principal_point=None,
                 distortion=None, pose=None):
        super(CameraModel, self).__init__(chip_size, focal_length, principal_point,
            distortion, pose)



    def snap(self, mesh, shaders=None):
        """ Takes image of mesh using camera
        :return:
            - depth_image - Depth image of scene, pixels seeing no object are set to NaN
            - color_image - Color image (RGB) of scene, pixels seeing no object are set to NaN
            - P - Scene points (only valid points)
        """
        # Generate camera rays and triangles
        rayorig = self._pose.get_translation()
        img = np.ones((self._chip_size[1], self._chip_size[0]))
        raydirs = self.depth_image_to_scene_points(img) - rayorig
        # Run ray tracer
        rt = RayTracer(rayorig, raydirs, mesh.vertices, mesh.triangles)
        rt.run()
        # Calculate shading
        if shaders is None:
            # Default shader is a point light directly at the position of the camera
            shaders = [ ShaderPointLight(self._pose.get_translation())]
        # Run all shaders and sum up RGB values from each shader
        P = rt.get_points_cartesic()
        C = np.zeros_like(P)
        for shader in shaders:
            C += shader.run(self, rt, mesh)
        # Finally clip to values in [0..1]
        C = np.clip(C, 0.0, 1.0)
        # Determine color and depth images
        depth_image, color_image = self.scene_points_to_depth_image(P, C)
        # Point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(P)
        pcl.colors = o3d.utility.Vector3dVector(C)
        return depth_image, color_image, pcl
