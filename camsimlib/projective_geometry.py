""" Mathematics for projective geometry to implement a camera or a projector
"""

import copy
import json
import numpy as np
import open3d as o3d

from abc import ABC, abstractmethod
from camsimlib.rays import Rays
from camsimlib.image_mapping import image_points_to_indices, image_indices_to_points, \
    image_points_on_chip_mask, image_indices_on_chip_mask
from camsimlib.lens_distortion_model import LensDistortionModel
from trafolib.trafo3d import Trafo3d



class ProjectiveGeometry(ABC):
    """ Base class providing mathematics for projective geometry
    for implementing a camera or a projector

    Coordinate system with Z axis pointing into direction of view

    Z               X - Axis
                    self.get_chip_size()[0] = width
      X--------->   depth_image.shape[1]
      |
      |
      |
      |
      V

    Y - Axis
    self.get_chip_size()[1] = height
    depth_image.shape[0]

    """

    def __init__(self, focal_length=100, principal_point=None,
                 distortion=None, pose=None):
        """ Constructor
        :param focal_length: See set_focal_length()
        :param principal_point: See set_principal_point(); if not provided,
            it is set center of chip
        :param distortion: See set_distortion()
        :param pose: See set_pose()
        """
        # focal_length
        self._focal_length = None
        self.set_focal_length(focal_length)
        # principal_point
        self._principal_point = None
        if principal_point is not None:
            self.set_principal_point(principal_point)
        else:
            self.set_principal_point(self.get_chip_size() / 2.0)
        # distortion
        self._distortion = LensDistortionModel()
        if distortion is not None:
            self.set_distortion(distortion)
        # pose: transformation from world to self
        self._pose = None
        if pose is None:
            self.set_pose(Trafo3d())
        else:
            self.set_pose(pose)



    def __str__(self):
        """ String representation of self
        :return: String representing of self
        """
        str = 'ProjectiveGeometry('
        with np.printoptions(precision=1, suppress=True):
            str += f'f={self._focal_length}, '
            str += f'c={self._principal_point}, '
        str += f'dist={self._distortion}, '
        str += f'pose={self._pose}'
        str += ')'
        return str



    def __copy__(self):
        """ Shallow copy
        :return: A shallow copy of self
        """
        return self.__class__(focal_length=self._focal_length,
                              principal_point=self._principal_point,
                              distortion=self._distortion,
                              pose=self._pose)



    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(focal_length=copy.deepcopy(self._focal_length, memo),
                                principal_point=copy.deepcopy(self._principal_point, memo),
                                distortion=copy.deepcopy(self._distortion.get_coefficients(), memo),
                                pose=copy.deepcopy(self._pose, memo))
        memo[id(self)] = result
        return result



    @abstractmethod
    def get_chip_size(self):
        """ Get chip size
        See set_chipsize().
        :return: Chip size
        """
        pass



    def set_focal_length(self, focal_length):
        """ Set focal length
        Focal length, either as scalar f or as vector (fx, fy)
        Unit for focal_length is pixels for both fx and fy. To convert this
        into a real world focal length in unit millimeters, you would have to
        assume a certain pixel density on the chip for conversion.
        :param focal_length: Focal length
        """
        flen = np.asarray(focal_length)
        if flen.size == 1:
            flen = np.append(flen, flen)
        elif flen.size > 2:
            raise ValueError('Provide 1d or 2d focal length')
        if np.any(flen < 0) or np.any(np.isclose(flen, 0)):
            raise ValueError('Provide positive focal length')
        self._focal_length = flen



    def get_focal_length(self):
        """ Get focal length
        See set_focal_length().
        :return: Focal length as vector (fx, fy)
        """
        return self._focal_length



    def set_principal_point(self, principal_point):
        """ Set principal point
        The principal point is the intersection point of optical axis with chip
        and is defined in pixels coordinates (cx, cy)
        Unit of principal point is pixels, for both cx and cy.
        :param principal_point: Principal point
        """
        ppoint = np.asarray(principal_point)
        if ppoint.size != 2:
            raise ValueError('Provide 2d principal point')
        self._principal_point = ppoint



    def get_principal_point(self):
        """ Get principal point
        See set_principal_point().
        :return: Principal point
        """
        return self._principal_point



    def set_camera_matrix(self, camera_matrix):
        """ Sets parameters focal lengths and principal point from camera matrix
        Camera matrix has to have shape 3x3,
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
        See set_camera_matrix().
        :return: Camera matrix
        """
        return np.array([
            [self._focal_length[0], 0.0, self._principal_point[0]],
            [0.0, self._focal_length[1], self._principal_point[1]],
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
        self._distortion.set_coefficients(distortion)



    def get_distortion(self):
        """ Get distortion coefficients
        See set_distortion().
        :return: Distortion parameters
        """
        return self._distortion.get_coefficients()



    def set_pose(self, pose):
        """ Set pose
        Transformation from world coordinate system to projective
        geometry (camera, projector) coordinate system as Trafo3d object.
        :param pose: 6d pose
        """
        self._pose = pose



    def place(self, point):
        """ Places projective geometry at a certain point
        :param point: Point to place the projective geometry at
        """
        P = np.asarray(point)
        if P.size != 3:
            raise ValueError('Provide 3d vector')
        self._pose.set_translation(P)



    def look_at(self, point):
        """ Rotates the projective geometry to look at certain point
        Rotates the so that the optical axis of the projective geometry
        (which direction is the z-axis of the projective geometry
        coordinate system) goes through this user specified point
        in the scene.
        :param point: Point in scene to look at
        """
        lookat_point = np.asarray(point)
        if lookat_point.size != 3:
            raise ValueError('Provide 3d vector')
        # Determine z unit vector e_z (forward)
        e_z = lookat_point - self._pose.get_translation()
        e_z_len = np.linalg.norm(e_z)
        if np.isclose(e_z_len, 0):
            raise ValueError('Point to look at is too close to center of projective geometry')
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
        self._pose.set_rotation_matrix(rotation_matrix)



    def roll(self, angle):
        """ Rotate projective geometry around the optical axis
        Rotates projective geometry around the optical axis (which
        is Z axis of the projective geometry coordinate system);
        positive direction is determined by right-hand-screw-rule
        :param angle: Rotation angle in radians
        """
        trafo = Trafo3d(rpy=(0, 0, angle))
        self._pose = self._pose * trafo



    def move_closer(self, distance):
        """ Moves projective geometry along the optical axis
        Moves projective geometry along the optical axis (which is
        the Z axis of the projective geometry coordinate system);
        positive distance is movement in positive Z direction
        (moving projective geometry closer to object)
        :param distance: Distance to move
        """
        rotation_matrix = self._pose.get_rotation_matrix()
        new_translation = self._pose.get_translation() + distance * rotation_matrix[:, 2]
        self._pose.set_translation(new_translation)



    def get_pose(self):
        """ Get pose
        Transformation from world coordinate system to projective geometry
        coordinate system as Trafo3d object
        :return: 6d pose
        """
        return self._pose



    def get_rvec_tvec(self):
        """ Get rvec and tvec as used in OpenCV
        :return: rvec (Rodrigues vector), tvec (translation)
        """
        return self._pose.get_rotation_rodrigues(), \
            self._pose.get_translation()



    def get_cs(self, size=1.0):
        """ Returns a representation of the coordinate system of the projective geometry
        Returns Open3d TriangleMesh object representing the coordinate system
        of the projective geometry that can be used for visualization
        :param size: Length of the coordinate axes of the coordinate system
        :return: Coordinate system as Open3d mesh object
        """
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coordinate_system.transform(self._pose.get_homogeneous_matrix())
        return coordinate_system



    def get_frustum(self, size=1.0, color=(0, 0, 0)):
        """ Returns a representation of the frustum of the projective geometry
        Returns Open3d LineSet object representing the frustum
        of the projective geometry that can be used for visualization.
        (A "frustum" is a cone with the top cut off.)
        :param size: Length of the sides of the frustum
        :param color: Color of the frustum
        :return: Frustum as Open3d mesh object
        """
        # Create image (chip) with all points NaN except corners
        dimg = np.zeros((self.get_chip_size()[1], self.get_chip_size()[0]))
        dimg[:] = np.NaN
        dimg[0, 0] = size
        dimg[0, -1] = size
        dimg[-1, 0] = size
        dimg[-1, -1] = size
        # Get the 3D points of the chip corner points
        P = self.depth_image_to_scene_points(dimg)
        # Create line-set visualizing frustum
        line_set = o3d.geometry.LineSet()
        points = np.vstack((self._pose.get_translation(), P))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(color, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set



    def json_save(self, filename):
        """ Save object parameters to json file
        :param filename: Filename of json file
        """
        params = {}
        self.dict_save(params)
        with open(filename, 'w') as file_handle:
            json.dump(params, file_handle, indent=4, sort_keys=True)



    def dict_save(self, param_dict):
        """ Save object parameters to dictionary
        :param param_dict: Dictionary to store projective geometry parameters in
        """
        param_dict['focal_length'] = self._focal_length.tolist()
        param_dict['principal_point'] = self._principal_point.tolist()
        self._distortion.dict_save(param_dict)
        param_dict['pose'] = {}
        self._pose.dict_save(param_dict['pose'])



    def json_load(self, filename):
        """ Load object parameters from json file
        :param filename: Filename of json file
        """
        with open(filename) as file_handle:
            params = json.load(file_handle)
        self.dict_load(params)



    def dict_load(self, param_dict):
        """ Load object parameters from dictionary
        :param param_dict: Dictionary with projective geometry parameters
        """
        self._focal_length = np.array(param_dict['focal_length'])
        self._principal_point = np.array(param_dict['principal_point'])
        self._distortion.dict_load(param_dict)
        self._pose.dict_load(param_dict['pose'])



    def calculate_opening_angles(self):
        """ Calculate opening angles
        Result is an estimate: distortions are not considered
        :return: Opening angles in x and y in radians
        """
        return 2.0 * np.arctan2(self.get_chip_size() / 2.0,
            self.get_focal_length())



    def scene_to_chip(self, P):
        """ Transforms points in scene to points on chip
        This function does not do any clipping or boundary checking;
        use points_on_chip_mask() to get mask of points on chip.
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :return: n points p=(u, v, d) on chip, shape (n, 3)
        """
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError('Provide scene coordinates of shape (n, 3)')
        # Transform points from world coordinate system to projective geometry coordinate system
        P = self._pose.inverse() * P
        # Mask points with Z lesser or equal zero
        valid = P[:, 2] > 0.0
        # projection
        pp = np.zeros((np.sum(valid), 2))
        pp[:, 0] = P[valid, 0] / P[valid, 2]
        pp[:, 1] = P[valid, 1] / P[valid, 2]
        # lens distortion
        pp = self._distortion.undistort(pp)
        # focal length and principal point
        p = np.empty_like(P)
        p[:] = np.NaN
        p[valid, 0] = self._focal_length[0] * pp[:, 0] + self._principal_point[0]
        p[valid, 1] = self._focal_length[1] * pp[:, 1] + self._principal_point[1]
        p[valid, 2] = np.linalg.norm(P[valid, :], axis=1)
        return p



    def points_on_chip_mask(self, points):
        return image_points_on_chip_mask(points, self.get_chip_size())



    def scene_points_to_depth_image(self, P, C=None):
        """ Transforms points in scene to depth image
        Image is initialized with np.NaN, invalid chip coordinates are filtered
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :param C: n colors C=(R, G, B) for each point; same shape as P; optional
        :return: Depth image, matrix of shape (self.get_chip_size()[1], self.get_chip_size()[0]),
            each element is distance; if C was provided, also returns color image
            of same size
        """
        p = self.scene_to_chip(P)
        indices = image_points_to_indices(p[:, 0:2])
        indices = np.round(indices).astype(int)
        on_chip_mask = image_indices_on_chip_mask(indices, self.get_chip_size())
        # Initialize empty image with NaN
        depth_image = np.empty((self.get_chip_size()[1], self.get_chip_size()[0]))
        depth_image[:] = np.NaN
        # Set image coordinates to distance values
        depth_image[indices[on_chip_mask, 0], indices[on_chip_mask, 1]] = p[on_chip_mask, 2]
        # If color values given, create color image as well
        if C is not None:
            if not np.array_equal(P.shape, C.shape):
                raise ValueError('P and C have to have the same shape')
            color_image = np.empty((self.get_chip_size()[1], self.get_chip_size()[0], 3))
            color_image[:] = np.NaN
            color_image[indices[on_chip_mask, 0], indices[on_chip_mask, 1], :] = C[on_chip_mask, :]
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
        pp[:, 0] = (p[:, 0] - self._principal_point[0]) / self._focal_length[0]
        pp[:, 1] = (p[:, 1] - self._principal_point[1]) / self._focal_length[1]
        # lens distortion
        pp = self._distortion.distort(pp)
        # projection
        P = np.zeros(p.shape)
        P[:, 2] = p[:, 2] / np.sqrt(np.sum(np.square(pp), axis=1) + 1.0)
        P[:, 0] = pp[:, 0] * P[:, 2]
        P[:, 1] = pp[:, 1] * P[:, 2]
        # Transform points from projective geometry coordinate system
        # to world coordinate system
        P = self._pose * P
        return P



    def depth_image_to_scene_points(self, img):
        """ Transforms depth image to list of scene points
        :param img: Depth image, matrix of shape (self.get_chip_size()[1], self.get_chip_size()[0]),
            each element is distance or NaN
        :return: n points P=(X, Y, Z) in scene, shape (n, 3) with
            n=np.prod(self.get_chip_size()) - number of NaNs
        """
        if self.get_chip_size()[0] != img.shape[1] or self.get_chip_size()[1] != img.shape[0]:
            raise ValueError('Provide depth image of proper size')
        mask = np.isfinite(img)
        if not np.all(img[mask] >= 0.0):
            raise ValueError('Depth image must contain only positive distances or NaN')
        # Pixel coordinates
        rows = np.arange(self.get_chip_size()[1])
        cols = np.arange(self.get_chip_size()[0])
        rows, cols = np.meshgrid(rows, cols, indexing='ij')
        indices = np.vstack((rows.flatten(), cols.flatten())).T
        p = image_indices_to_points(indices)
        p = np.hstack((p, img.flatten().reshape((-1, 1))))
        mask = np.isfinite(p[:, 2])
        return self.chip_to_scene(p[mask])



    def get_rays(self, points=None):
        """ Gets rays of camera object
        If no points are provided, all camera rays are returned. If user
        provided points, the rays just for those points are returned.
        The rays are defined in world coordinates and are normalized.
        :param points: n 2D points on the camera chip p(u,v), shape (n, 2)
        :return: Ray object with requested rays
        """
        if points is None:
            img = np.ones((self.get_chip_size()[1], self.get_chip_size()[0]))
            P = self.depth_image_to_scene_points(img)
        else:
            if points.ndim != 2 or points.shape[1] != 2:
                raise ValueError('Provide chip coordinates of shape (n, 2)')
            p = np.ones((points.shape[0], 3))
            p[:, 0:2] = points
            P = self.chip_to_scene(p)
        rayorigs = self._pose.get_translation()
        raydirs = P - rayorigs
        rays = Rays(rayorigs, raydirs)
        rays.normalize()
        return rays
