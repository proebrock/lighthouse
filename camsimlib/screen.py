import copy
import json
import numpy as np
import open3d as o3d

from trafolib.trafo3d import Trafo3d
from camsimlib.o3d_utils import mesh_generate_image



class Screen:
    """ A screen or monitor of real world dimensions displaying a colored image

                  shape[1]
                .------->
                |
       shape[0] |   .----------------.
                |   |                |
                V   |                |
                    |     Image      | height
                    |                |
                    |                |
                /   |                |
                |   .----------------.
              Y |          width
                |
                .------->
              Z      X

    :param dimensions:
    :param image_or_shape:
    :param pose:
    """

    def __init__(self, dimensions, image_or_shape, pose=None):
        """ Constructor
        :param dimensions: Real-world dimensions of the screen in millimeters (width, height)
        :param image_or_shape: Color image or shape of an image (height, width)
        """
        # dimensions
        self._dimensions = np.asarray(dimensions)
        if self._dimensions.size != 2:
            raise ValueError('Provide proper dimensions')
        # image
        ios = np.asarray(image_or_shape)
        if (ios.ndim == 1) and (ios.size == 2):
            self._image = np.zeros((ios[0], ios[1], 3), dtype=np.uint8)
        elif ios.ndim == 3:
            self._image = ios
        else:
            raise ValueError('Provide proper image or image shape')
        # pose
        if pose is None:
            self._pose = Trafo3d()
        else:
            self._pose = pose



    def __str__(self):
        """ String representation of self
        :return: String representing of self
        """
        return ('Screen(' +
            f'dim={self._dimensions}, ' +
            f'image.shape={self._image.shape}, ' +
            f'pose={self._pose}' +
            ')'
            )



    def __copy__(self):
        """ Shallow copy
        :return: A shallow copy of self
        """
        return self.__class__(dimensions=self._dimensions,
                              image_or_shape=self._image,
                              pose=self._pose)



    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(dimensions=copy.deepcopy(self._dimensions, memo),
                                image_or_shape=copy.deepcopy(self._image, memo),
                                pose=copy.deepcopy(self._pose, memo))
        memo[id(self)] = result
        return result



    def get_dimensions(self):
        """ Get real-world dimensions of the screen in millimeters
        :return: Image
        """
        return self._dimensions



    def set_image(self, image):
        """ Set image displayed by screen
        Image must have same shape as originally configured with the constructor
        :param image: Image
        """
        if not np.all(image.shape == self._image.shape):
            raise ValueError('Provide image of same dimensions')
        self._image = image



    def get_image(self):
        """ Get image displayed by screen
        :return: Image
        """
        return self._image



    def set_pose(self, pose):
        """ Set pose
        Transformation from world coordinate system to screen as Trafo3d object
        :param pose: 6d pose
        """
        self._pose = pose



    def get_pose(self):
        """ Get pose
        Transformation from world coordinate system to screen as Trafo3d object
        :return: 6d pose
        """
        return self._pose



    def get_cs(self, size=1.0):
        """ Returns a representation of the coordinate system of the screen
        Returns Open3d TriangleMesh object representing the coordinate system
        of the screen that can be used for visualization
        :param size: Length of the coordinate axes of the coordinate system
        :return: Coordinate system as Open3d mesh object
        """
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coordinate_system.transform(self._pose.get_homogeneous_matrix())
        return coordinate_system



    def get_mesh(self):
        """ Returns a visual representation of the screen
        Returns Open3d TriangleMesh object representing the screen
        that can be used for visualization
        :return: Visualization as Open3d mesh object
        """
        pixel_sizes = (self._dimensions[0] / self._image.shape[1],
            self._dimensions[1] / self._image.shape[0])
        mesh = mesh_generate_image(self._image, pixel_sizes)
        mesh.transform(self._pose.get_homogeneous_matrix())
        return mesh



    def json_save(self, filename):
        """ Save screen parameters to json file
        :param filename: Filename of json file
        """
        params = {}
        self.dict_save(params)
        with open(filename, 'w') as file_handle:
            json.dump(params, file_handle, indent=4, sort_keys=True)



    def dict_save(self, param_dict):
        """ Save screen parameters to dictionary
        :param param_dict: Dictionary to store parameters in
        """
        param_dict['dimensions'] = self._dimensions.tolist()
        param_dict['image_shape'] = (self._image.shape[0], self._image.shape[1])
        param_dict['pose'] = {}
        self._pose.dict_save(param_dict['pose'])



    def json_load(self, filename):
        """ Load screen parameters from json file
        :param filename: Filename of json file
        """
        with open(filename) as file_handle:
            params = json.load(file_handle)
        self.dict_load(params)



    def dict_load(self, param_dict):
        """ Load screen parameters from dictionary
        :param param_dict: Dictionary with screen parameters
        """
        self._dimensions = np.array(param_dict['dimensions'])
        image_shape = np.array(param_dict['image_shape'])
        self._image = np.zeros((image_shape[0], image_shape[1], 3), dtype=np.uint8)
        self._pose.dict_load(param_dict['pose'])



    def screen_to_scene(self, p, check_for_valid=True):
        """ Converts 2D screen points to 3D scene points (world coordinate system)
        :param p: 2D screen points, shape (n, 2), type float (subpixels allowed)
        :param check_for_valid: True if checks for valid screen coordinates desired
        :return: 3D scene points, shape (n, 3)
        """
        if p.shape[1] != 2:
            raise ValueError('Provide proper dimensions')
        P = np.zeros((p.shape[0], 3))
        P[:, 0] = (self._dimensions[0] * (p[:, 1] + 0.5)) / self._image.shape[1]
        P[:, 1] = (self._dimensions[1] * (p[:, 0] + 0.5)) / self._image.shape[0]
        P[:, 1] = self._dimensions[1] - P[:, 1]
        if check_for_valid:
            valid_screen_points_mask = np.logical_and.reduce((
                P[:, 0] >= 0.0,
                P[:, 0] <= self._dimensions[0],
                P[:, 1] >= 0.0,
                P[:, 1] <= self._dimensions[1],
            ))
            if sum(~valid_screen_points_mask) > 0:
                raise ValueError('Provide valid points on screen')
        # Transform points from screen coordinate system
        # to world coordinate system
        P = self._pose * P
        return P
