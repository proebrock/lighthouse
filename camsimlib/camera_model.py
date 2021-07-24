# -*- coding: utf-8 -*-
""" Simulation of depth and/or RGB cameras
"""

import copy
import json
import numpy as np
import open3d as o3d

from camsimlib.lens_distortion_model import LensDistortionModel
from camsimlib.ray_tracer import RayTracer
from trafolib.trafo3d import Trafo3d



class CameraModel:
    """ Class for simulating a depth and/or RGB camera

    Camera coordinate system with Z-Axis pointing into direction of view

    Z               X - Axis
                    self.chip_size[0] = width
      X--------->   depth_image second dimension
      |
      |
      |
      |
      V

    Y - Axis
    self.chip_size[1] = height
    depth_image first dimension

    """

    def __init__(self, chip_size=(40, 30), focal_length=100, principal_point=None,
                 distortion=None, camera_pose=None,
                 shading_mode='gouraud'):
        """ Constructor
        :param chip_size: See set_chip_size()
        :param focal_length: See set_focal_length()
        :param principal_point: See set_principal_point(); if not provided, it is set center of chip
        :param distortion: See set_distortion()
        :param camera_pose: See set_camera_pose()
        :param shading_mode: Shading mode, 'flat' or 'gouraud'
        """
        # chip_size
        self.chip_size = None
        self.set_chip_size(chip_size)
        # focal_length
        self.focal_length = None
        self.set_focal_length(focal_length)
        # principal_point
        self.principal_point = None
        if principal_point is not None:
            self.set_principal_point(principal_point)
        else:
            self.set_principal_point(self.chip_size / 2.0)
        # distortion
        self.distortion = LensDistortionModel()
        if distortion is not None:
            self.set_distortion(distortion)
        # camera position: transformation from world to camera
        self.camera_pose = None
        if camera_pose is None:
            self.set_camera_pose(Trafo3d())
        else:
            self.set_camera_pose(camera_pose)
        # shading mode
        if shading_mode not in ('flat', 'gouraud'):
            raise ValueError(f'Unknown shading mode "{shading_mode}')
        self.shading_mode = shading_mode



    def __str__(self):
        """ String representation of camera object
        :return: String representing camera object
        """
        return (f'chip_size={self.chip_size}, '
                f'f={self.focal_length}, '
                f'c={self.principal_point}, '
                f'distortion={self.distortion}, '
                f'camera_pose={self.camera_pose}')



    def __copy__(self):
        """ Shallow copy
        :return: A shallow copy of self
        """
        return self.__class__(chip_size=self.chip_size,
                              focal_length=self.focal_length,
                              principal_point=self.principal_point,
                              distortion=self.distortion,
                              camera_pose=self.camera_pose,
                              shading_mode=self.shading_mode)



    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(chip_size=copy.deepcopy(self.chip_size, memo),
                                focal_length=copy.deepcopy(self.focal_length, memo),
                                principal_point=copy.deepcopy(self.principal_point, memo),
                                distortion=copy.deepcopy(self.distortion.get_coefficients(), memo),
                                camera_pose=copy.deepcopy(self.camera_pose, memo),
                                shading_mode=copy.deepcopy(self.shading_mode, memo))
        memo[id(self)] = result
        return result



    def set_chip_size(self, chip_size):
        """ Set chip size
        The size of the camera chip in pixels, width x height
        :param chip_size: Chip size
        """
        csize = np.asarray(chip_size, dtype=np.int64)
        if csize.size != 2:
            raise ValueError('Provide 2d chip size in pixels')
        if np.any(csize < 1):
            raise ValueError('Provide positive chip size')
        self.chip_size = csize



    def get_chip_size(self):
        """ Get chip size
        The size of the camera chip in pixels, width x height
        :return: Chip size
        """
        return self.chip_size



    def set_focal_length(self, focal_length):
        """ Set focal length
        Focal length, either as scalar f or as vector (fx, fy)
        :param focal_length: Focal length
        """
        flen = np.asarray(focal_length)
        if flen.size == 1:
            flen = np.append(flen, flen)
        elif flen.size > 2:
            raise ValueError('Provide 1d or 2d focal length')
        if np.any(flen < 0) or np.any(np.isclose(flen, 0)):
            raise ValueError('Provide positive focal length')
        self.focal_length = flen



    def get_focal_length(self):
        """ Get focal length
        :return: Focal length as vector (fx, fy)
        """
        return self.focal_length



    def set_principal_point(self, principal_point):
        """ Set principal point
        The principal point is the intersection point of optical axis with chip
        and is defined in pixels coordinates (cx, cy)
        :param principal_point: Principal point
        """
        ppoint = np.asarray(principal_point)
        if ppoint.size != 2:
            raise ValueError('Provide 2d principal point')
        self.principal_point = ppoint



    def get_principal_point(self):
        """ Get principal point
        The principal point is the intersection point of optical axis with chip
        and is defined in pixels coordinates (cx, cy)
        :return: Principal point
        """
        return self.principal_point



    def set_camera_matrix(self, camera_matrix):
        """ Sets parameters focal lengths and principal point from camera matrix
        :param camera_matrix: Camera matrix
        """
        if camera_matrix.ndim != 2 or \
            camera_matrix.shape[0] != 3 or \
            camera_matrix.shape[1] != 3:
            raise Exception('Provide camera matrix of correct dimensions')
        if camera_matrix[0, 1] != 0 or \
            camera_matrix[1, 0] != 0 or \
            camera_matrix[2, 0] != 0 or \
            camera_matrix[2, 1] != 0 or \
            camera_matrix[2, 2] != 1:
            raise Exception('Unexpected values found in camera matrix')
        self.set_focal_length((camera_matrix[0, 0], camera_matrix[1, 1]))
        self.set_principal_point((camera_matrix[0, 2], camera_matrix[1, 2]))



    def get_camera_matrix(self):
        """ Get camera matrix
        Returns 3x3 camera matrix containing focal lengths and principal point
        :return: Camera matrix
        """
        return np.array([
            [self.focal_length[0], 0.0, self.principal_point[0]],
            [0.0, self.focal_length[1], self.principal_point[1]],
            [0.0, 0.0, 1.0]
            ])


    def set_distortion(self, distortion):
        """ Set distortion coefficients

        Lens distortion coefficients according to OpenCV model:
        (0,  1,  2,  3,   4,   5,  6,  7,   8,  9,  10  11)    <- Indices in distortion
        (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]) <- OpenCV names
        k1-k6 Radial distortion coefficients
        p1-p2 Tangential distortion coefficients
        s1-s4 Thin prism distortion coefficients

        :param distortion: Distortion coefficients
        """
        self.distortion.set_coefficients(distortion)



    def get_distortion(self):
        """ Get distortion coefficients
        :return: Distortion parameters
        """
        return self.distortion.get_coefficients()



    def set_camera_pose(self, camera_pose):
        """ Set camera pose
        Transformation from world coordinate system to camera coordinate system as Trafo3d object
        :param camera_pose: Camera position
        """
        self.camera_pose = camera_pose



    def place_camera(self, point):
        """ Places camera at a certain point
        :param point: Point to place the camera at
        """
        camera_point = np.asarray(point)
        if camera_point.size != 3:
            raise ValueError('Provide 3d vector')
        self.camera_pose.set_translation(camera_point)



    def look_at(self, point):
        """ Rotates the camera to look at certain point
        Rotates the camera so that the optical axis of the camera
        (which direction is the z-axis of the camera coordinate system)
        goes through this user specified point in the scene.
        :param point: Point in scene to look at
        """
        lookat_point = np.asarray(point)
        if lookat_point.size != 3:
            raise ValueError('Provide 3d vector')
        # Determine z unit vector e_z (forward)
        e_z = lookat_point - self.camera_pose.get_translation()
        e_z_len = np.linalg.norm(e_z)
        if np.isclose(e_z_len, 0):
            raise ValueError('Point to look at is too close to camera')
        e_z = e_z / e_z_len
        # Determine x unit vector e_x (right)
        tmp = np.array([0, 0, 1])
        e_x = np.cross(e_z, tmp)
        e_x_len = np.linalg.norm(e_x)
        if np.isclose(e_x_len, 0):
            # tmp and e_z are parallel or anti-parallel,
            # so set e_x to e_x of world coordinate system
            e_x = np.array([1, 0, 0])
        else:
            e_x = e_x / e_x_len
        # Determine y unit vector e_y (down)
        e_y = np.cross(e_z, e_x)
        e_y = e_y / np.linalg.norm(e_y)
        # Set rotation matrix
        rotation_matrix = np.array((e_x, e_y, e_z)).T
        self.camera_pose.set_rotation_matrix(rotation_matrix)



    def roll_camera(self, angle):
        """ Rotate camera around the optical axis
        Rotates camera around the optical axis (which is Z axis of the
        camera coordinate system); positive direction is determined
        by right-hand-screw-rule
        :param angle: Rotation angle in radians
        """
        trafo = Trafo3d(rpy=(0, 0, angle))
        self.camera_pose = self.camera_pose * trafo



    def move_camera_closer(self, distance):
        """ Moves camera along the optical axis
        Moves camera along the optical axis (which is the Z axis of the
        camera coordinate system); positive distance is movement in
        positive Z direction (moving camera closer to object)
        :param distance: Distance to move
        """
        rotation_matrix = self.camera_pose.get_rotation_matrix()
        new_translation = self.camera_pose.get_translation() + distance * rotation_matrix[:, 2]
        self.camera_pose.set_translation(new_translation)



    def get_camera_pose(self):
        """ Get camera pose
        Transformation from world coordinate system to camera coordinate system as Trafo3d object
        :return: Camera position
        """
        return self.camera_pose



    def json_save(self, filename):
        """ Save camera parameters to json file
        :param filename: Filename of json file
        """
        params = {}
        self.dict_save(params)
        with open(filename, 'w') as file_handle:
            json.dump(params, file_handle, indent=4, sort_keys=True)



    def dict_save(self, param_dict):
        """ Save camera parameters to dictionary
        :param params: Dictionary to store camera parameters in
        """
        param_dict['chip_size'] = self.chip_size.tolist()
        param_dict['focal_length'] = self.focal_length.tolist()
        param_dict['principal_point'] = self.principal_point.tolist()
        self.distortion.dict_save(param_dict)
        param_dict['camera_pose'] = {}
        param_dict['camera_pose']['t'] = self.camera_pose.get_translation().tolist()
        param_dict['camera_pose']['q'] = self.camera_pose.get_rotation_quaternion().tolist()



    def json_load(self, filename):
        """ Load camera parameters from json file
        :param filename: Filename of json file
        """
        with open(filename) as file_handle:
            params = json.load(file_handle)
        self.dict_load(params)



    def dict_load(self, param_dict):
        """ Load camera parameters from dictionary
        :param params: Dictionary with camera parameters
        """
        self.chip_size = np.array(param_dict['chip_size'])
        self.focal_length = np.array(param_dict['focal_length'])
        self.principal_point = np.array(param_dict['principal_point'])
        self.distortion.dict_load(param_dict)
        self.camera_pose = Trafo3d(t=param_dict['camera_pose']['t'],
                                   q=param_dict['camera_pose']['q'])



    def calculate_opening_angles(self):
        """ Calculate opening angles
        :return: Opening angles in x and y in radians
        """
        p = np.array([[self.chip_size[1], self.chip_size[0], 1]])
        P = self.chip_to_scene(p)
        return 2.0 * np.arctan2(P[0, 0], P[0, 2]), \
            2.0 * np.arctan2(P[0, 1], P[0, 2])



    def scale_resolution(self, factor=1.0):
        """ Scale camera resolution
        The camera resolution heavily influences the computational resources needed
        to snap images. So for most setups it makes sense to keep a low resolution
        camera to take test images and then later to scale up the camera resolution.
        This method scales chip_size, f, c and distortion accordingly to increase
        (factor > 1) or reduce (factor > 1) camera resolution.
        :param factor: Scaling factor
        """
        self.chip_size = (factor * self.chip_size).astype(np.int64)
        self.focal_length = factor * self.focal_length
        self.principal_point = factor * self.principal_point



    def scene_to_chip(self, P):
        """ Transforms points in scene to points on chip
        This function does not do any clipping or boundary checking!
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :return: n points p=(u, v, d) on chip, shape (n, 3)
        """
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError('Provide scene coordinates of shape (n, 3)')
        # Transform points from world coordinate system to camera coordinate system
        P = self.camera_pose.inverse() * P
        # Mask points with Z lesser or equal zero
        valid = P[:, 2] > 0.0
        # projection
        pp = np.zeros((np.sum(valid), 2))
        pp[:, 0] = P[valid, 0] / P[valid, 2]
        pp[:, 1] = P[valid, 1] / P[valid, 2]
        # lens distortion
        pp = self.distortion.undistort(pp)
        # focal length and principal point
        p = np.NaN * np.zeros(P.shape)
        p[valid, 0] = self.focal_length[0] * pp[:, 0] + self.principal_point[0]
        p[valid, 1] = self.focal_length[1] * pp[:, 1] + self.principal_point[1]
        p[valid, 2] = np.linalg.norm(P[valid, :], axis=1)
        return p



    def scene_points_to_depth_image(self, P, C=None):
        """ Transforms points in scene to depth image
        Image is initialized with np.NaN, invalid chip coordinates are filtered
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :param C: n colors C=(R, G, B) for each point; same shape as P; optional
        :return: Depth image, matrix of shape (self.chip_size[1], self.chip_size[0]),
            each element is distance; if C was provided, also returns color image
            of same size
        """
        p = self.scene_to_chip(P)
        # Clip image indices to valid points (can cope with NaN values in p)
        indices = np.round(p[:, 0:2]).astype(int)
        x_valid = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < self.chip_size[0])
        y_valid = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < self.chip_size[1])
        valid = np.logical_and(x_valid, y_valid)
        # Initialize empty image with NaN
        depth_image = np.NaN * np.empty((self.chip_size[1], self.chip_size[0]))
        # Set image coordinates to distance values
        depth_image[indices[valid, 1], indices[valid, 0]] = p[valid, 2]
        # If color values given, create color image as well
        if C is not None:
            if not np.array_equal(P.shape, C.shape):
                raise ValueError('P and C have to have the same shape')
            color_image = np.NaN * np.empty((self.chip_size[1], self.chip_size[0], 3))
            color_image[indices[valid, 1], indices[valid, 0], :] = C[valid, :]
            return depth_image, color_image
        return depth_image



    def chip_to_scene(self, p):
        """ Transforms points on chip to points in scene
        This function does not do any clipping boundary checking!
        :param p: n points p=(u, v, d) on chip, shape (n, 3)
        :return: n points P=(X, Y, Z) in scene, shape (n, 3)
        """
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Provide chip coordinates of shape (n, 3)')
        # focal length and principal point
        pp = np.zeros((p.shape[0], 2))
        pp[:, 0] = (p[:, 0] - self.principal_point[0]) / self.focal_length[0]
        pp[:, 1] = (p[:, 1] - self.principal_point[1]) / self.focal_length[1]
        # lens distortion
        pp = self.distortion.distort(pp)
        # projection
        P = np.zeros(p.shape)
        P[:, 2] = p[:, 2] / np.sqrt(np.sum(np.square(pp), axis=1) + 1.0)
        P[:, 0] = pp[:, 0] * P[:, 2]
        P[:, 1] = pp[:, 1] * P[:, 2]
        # Transform points from camera coordinate system to world coordinate system
        P = self.camera_pose * P
        return P



    def depth_image_to_scene_points(self, img):
        """ Transforms depth image to list of scene points
        :param img: Depth image, matrix of shape (self.chip_size[1], self.chip_size[0]),
            each element is distance or NaN
        :return: n points P=(X, Y, Z) in scene, shape (n, 3) with
            n=np.prod(self.chip_size) - number of NaNs
        """
        if self.chip_size[0] != img.shape[1] or self.chip_size[1] != img.shape[0]:
            raise ValueError('Provide depth image of proper size')
        mask = ~np.isnan(img)
        if not np.all(img[mask] >= 0.0):
            raise ValueError('Depth image must contain only positive distances or NaN')
        x = np.arange(self.chip_size[0])
        y = np.arange(self.chip_size[1])
        Y, X = np.meshgrid(y, x, indexing='ij')
        p = np.vstack((X.flatten(), Y.flatten(), img.flatten())).T
        mask = np.logical_not(np.isnan(p[:, 2]))
        return self.chip_to_scene(p[mask])



    @staticmethod
    def __flat_shading(mesh, P, triangle_idx, light_position):
        """ Calculate flat shading for multiple triangles
        We assume a point light source at light_position. For each
        triangle we calculate the dot product of the triangle normal and
        the vector from the vertex to the light source (normalized).
        This gives the intensity for this triangle.
        There is no coloring in this implementation of flat shading;
        Open3d has vertex colors but no triangle colors.
        :param mesh: Mesh of type MeshObject
        :param P: Intersection points on triangles in
            Cartesian coordinates (X, Y, Z), shape (n, 3)
        :param triangle_idx: Indices of n triangles whose shading we want to calculate, shape (n, )
        :param light_position: Position of the light
        :return: Shades of triangles; shape (n, 3) (RGB) [0.0..1.0]
        """
        triangle_normals = np.asarray(mesh.triangle_normals)[triangle_idx, :]
        # lightvec goes from intersection point to light source
        lightvecs = -P + light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=1)[:, np.newaxis]
        # Dot product of triangle_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        intensities = np.sum(triangle_normals * lightvecs, axis=1)
        # There are no colors so stack intensity 3 times to get RGB
        return np.vstack((intensities, intensities, intensities)).T



    @staticmethod
    def __gouraud_shading(mesh, Pbary, triangle_idx, light_position):
        """ Calculate the Gouraud shading for multiple points
        We assume a point light source at light_position. For each
        vertex of the triangle, we calculate the dot product of the
        vertex normal and the vector from the vertex to the light source
        (normalized). This gives the intensity for this vertex. Together
        with the vertex color we can determine the vertex color shade of
        each vertex. Those three color shades are then interpolated using
        the barycentric coordinate of the ray-triangle-intersection.
        :param mesh: Mesh of the scene (of type MeshObject)
        :param Pbary: Intersection points on triangles in
            barycentric coordinates (1-u-v, u, v), shape (n, 3)
        :param triangle_idx: Indices of n triangles whose shading we want to calculate, shape (n, )
        :param light_position: Position of the light
        :return: Shades of triangles; shape (n, 3) (RGB) [0.0..1.0]
        """
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(mesh.triangles)[triangle_idx, :]
        vertices = np.asarray(mesh.vertices)[triangles]
        vertex_normals = np.asarray(mesh.vertex_normals)[triangles]
        # lightvec goes from vertex to light source
        lightvecs = -vertices + light_position
        lightvecs /= np.linalg.norm(lightvecs, axis=2)[:, :, np.newaxis]
        # Dot product of vertex_normals and lightvecs; if angle between
        # those is 0°, the intensity is 1; the intensity decreases up
        # to an angle of 90° where it is 0
        vertex_intensities = np.sum(vertex_normals * lightvecs, axis=2)
        vertex_intensities = np.clip(vertex_intensities, 0.0, 1.0)
        # From intensity determine color shade
        if mesh.has_vertex_colors():
            vertex_colors = np.asarray(mesh.vertex_colors)[triangles]
        else:
            vertex_colors = np.ones((triangles.shape[0], 3, 3))
        vertex_color_shades = vertex_colors * vertex_intensities[:, :, np.newaxis]
        # Interpolate to get color of intersection point
        return np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)



    def snap(self, mesh):
        """ Takes image of mesh using camera
        :return:
            - depth_image - Depth image of scene, pixels seeing no object are set to NaN
            - color_image - Color image (RGB) of scene, pixels seeing no object are set to NaN
            - P - Scene points (only valid points)
        """
        # Generate camera rays and triangles
        rayorig = self.camera_pose.get_translation()
        img = np.ones((self.chip_size[1], self.chip_size[0]))
        raydirs = self.depth_image_to_scene_points(img) - rayorig
        triangles = np.asarray(mesh.vertices)[np.asarray(mesh.triangles)]
        # Run ray tracer
        rt = RayTracer(rayorig, raydirs, triangles)
        rt.run()
        P = rt.get_points_cartesic()
        Pbary = rt.get_points_barycentric()
        triangle_idx = rt.get_triangle_indices()
        # Calculate shading: Assume light position is position of the camera
        if self.shading_mode == 'flat':
            C = CameraModel.__flat_shading(mesh, P, triangle_idx, rayorig)
        elif self.shading_mode == 'gouraud':
            C = CameraModel.__gouraud_shading(mesh, Pbary, triangle_idx, rayorig)
        # Determine color and depth images
        depth_image, color_image = self.scene_points_to_depth_image(P, C)
        # Point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(P)
        pcl.colors = o3d.utility.Vector3dVector(C)
        return depth_image, color_image, pcl



    def get_cs(self, size=1.0):
        """ Returns a representation of the coordinate system of the camera
        Returns Open3d TriangleMesh object representing the coordinate system
        of the camera that can be used for visualization
        :param size: Length of the coordinate axes of the coordinate system
        :return: Coordinate system as Open3d mesh object
        """
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coordinate_system.transform(self.camera_pose.get_homogeneous_matrix())
        return coordinate_system



    def get_frustum(self, size=1.0, color=(0, 0, 0)):
        """ Returns a representation of the frustum of the camera
        Returns Open3d LineSet object representing the frustum
        of the camera that can be used for visualization.
        (A "frustum" is a cone with the top cut off.)
        :param size: Length of the sides of the frustum
        :param color: Color of the frustum
        :return: Frustum as Open3d mesh object
        """
        # Create image (chip) with all points NaN except corners
        dimg = np.zeros((self.chip_size[1], self.chip_size[0]))
        dimg[:] = np.NaN
        dimg[0, 0] = size
        dimg[0, -1] = size
        dimg[-1, 0] = size
        dimg[-1, -1] = size
        # Get the 3D points of the chip corner points
        P = self.depth_image_to_scene_points(dimg)
        # Create line-set visualizing frustum
        line_set = o3d.geometry.LineSet()
        points = np.vstack((self.camera_pose.get_translation(), P))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(color, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set
