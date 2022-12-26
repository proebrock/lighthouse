# -*- coding: utf-8 -*-
""" Simulation of depth and/or RGB cameras
"""

import copy
import numpy as np
import open3d as o3d

from camsimlib.projective_geometry import ProjectiveGeometry
from camsimlib.ray_tracer_embree import RayTracerEmbree as RayTracer
from camsimlib.shader_point_light import ShaderPointLight



class CameraModel(ProjectiveGeometry):
    """ Class for simulating a depth and/or RGB camera

    Camera coordinate system with Z-Axis pointing into direction of view

    Z               X - Axis
                    self.get_chip_size()[0] = width
      X--------->   depth_image second dimension
      |
      |
      |
      |
      V

    Y - Axis
    self.get_chip_size()[1] = height
    depth_image first dimension

    """
    def __init__(self, chip_size=(40, 30), focal_length=100, principal_point=None,
                 distortion=None, pose=None):
        self._chip_size = None
        self.set_chip_size(chip_size)
        super(CameraModel, self).__init__(focal_length, principal_point,
            distortion, pose)



    def __str__(self):
        """ String representation of self
        :return: String representing of self
        """
        return (super().__str__() +
                'CameraModel(' +
                f'chip_size={self._chip_size}, ' +
                ')'
                )



    def __copy__(self):
        """ Shallow copy
        :return: A shallow copy of self
        """
        return self.__class__(chip_size=self._chip_size,
                              focal_length=self._focal_length,
                              principal_point=self._principal_point,
                              distortion=self._distortion,
                              pose=self._pose)



    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(chip_size=copy.deepcopy(self._chip_size),
                                focal_length=copy.deepcopy(self._focal_length, memo),
                                principal_point=copy.deepcopy(self._principal_point, memo),
                                distortion=copy.deepcopy(self._distortion.get_coefficients(), memo),
                                pose=copy.deepcopy(self._pose, memo))
        memo[id(self)] = result
        return result



    def set_chip_size(self, chip_size):
        """ Set chip size
        The size of the chip in pixels, width x height
        Unit for chip size is pixels for both width and height.
        :param chip_size: Chip size
        """
        csize = np.asarray(chip_size, dtype=int)
        if csize.size != 2:
            raise ValueError('Provide 2d chip size in pixels')
        if np.any(csize < 1):
            raise ValueError('Provide positive chip size')
        self._chip_size = csize



    def get_chip_size(self):
        """ Get chip size
        See set_chipsize().
        :return: Chip size
        """
        return self._chip_size



    def scale_resolution(self, factor=1.0):
        """ Scale projective geometry resolution
        The projective geometry resolution heavily influences the computational
        resources needed e.g. to snap images. So for most setups it makes sense
        to keep a low resolution projective geometry to take test/preview images
        and then later to scale up the projective geometry resolution.
        This method scales chip_size, f, c and distortion accordingly to increase
        (factor > 1) or reduce (factor > 1) projective geometry resolution.
        :param factor: Scaling factor
        """
        self.set_chip_size((factor * self.get_chip_size()).astype(int))
        self.set_focal_length(factor * self.get_focal_length())
        self.set_principal_point(factor * self.get_principal_point())



    def dict_save(self, param_dict):
        """ Save camera model parameters to dictionary
        :param params: Dictionary to store projective geometry parameters in
        """
        super().dict_save(param_dict)
        param_dict['chip_size'] = self._chip_size.tolist()



    def dict_load(self, param_dict):
        """ Load camera model parameters from dictionary
        :param params: Dictionary with projective geometry parameters
        """
        super().dict_load(param_dict)
        self._chip_size = np.array(param_dict['chip_size'])



    def snap(self, mesh, shaders=None):
        """ Takes image of mesh using camera
        :return:
            - depth_image - Depth image of scene, pixels seeing no object are set to NaN
            - color_image - Color image (RGB) of scene, pixels seeing no object are set to NaN
            - P - Scene points (only valid points)
        """
        # Generate camera rays and triangles
        rayorig = self._pose.get_translation()
        img = np.ones((self.get_chip_size()[1], self.get_chip_size()[0]))
        raydirs = self.depth_image_to_scene_points(img) - rayorig
        # Run raytracer: we do "eye-based path tracing" starting from a
        # a ray from each camera pixel until we hit an object
        rt = RayTracer(rayorig, raydirs, [ mesh ])
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
