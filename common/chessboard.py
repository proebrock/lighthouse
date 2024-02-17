import sys
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_show
from common.mesh_utils import mesh_generate_image
from camsimlib.screen import Screen
from camsimlib.camera_model import CameraModel



class Chessboard:
    """ Representation of a chessboard usable for calibration.

                 Z         X, squares[0]
                    X --------->
                    |
          Y         |    .------------.
        squares[1]  |    |            |
                    |    |            |
                    V    |   Board    |
                         |            |
                         |            |
                         .------------.
    """

    def __init__(self, squares=(5, 6), square_length_pix=80,
        square_length_mm=20.0, pose=Trafo3d()):
        """ Constructor
        :param squares: Number of squares: width x height
        :param square_length_pix: Length of single square in pixels
        :param square_length_mm: Length of single square in millimeters
        :param pose: Transformation from world to CharucoBoard
        """
        self._squares = np.asarray(squares)
        assert self._squares.size == 2
        # One dimension has to be odd, the other even to
        # have a rotation invariant chessboard
        assert np.sum(self._squares % 2) == 1
        self._square_length_pix = square_length_pix
        self._square_length_mm = square_length_mm
        self._pose = pose



    def __str__(self):
        """ Get readable string representation of object
        :return: String representation of object
        """
        param_dict = {}
        self.dict_save(param_dict)
        return str(param_dict)



    def get_pose(self):
        """ Get transformation from world to center of MultiAruco object
        :return: Pose as Trafo3d object
        """
        return self._pose



    def set_pose(self, pose):
        """ Set transformation from world to center of MultiAruco object
        :param pose: Pose as Trafo3d object
        """
        self._pose = pose



    def get_cs(self, size):
        """ Get coordinate system object representing pose of MultiMarker object
        :param size: Length of coordinate axes
        :return: Coordinate system as Open3D mesh object
        """
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        cs.transform(self._pose.get_homogeneous_matrix())
        return cs



    def get_size_pix(self):
        """ Get size of board in pixels
        :return: Size as tupel in (X, Y)
        """
        return self._squares * self._square_length_pix



    def get_size_mm(self):
        """ Get size of board in millimeters
        :return: Size as tupel in (X, Y)
        """
        return self._squares * self._square_length_mm



    def get_pixelsize_mm(self):
        """ Get size of a single pixel in millimeters
        :return: Size as a scalar
        """
        return self._square_length_mm / self._square_length_pix



    def get_resolution_dpi(self):
        """ Get resolution of board in DPI (dots per inch)
        Use this resolution to print the board on paper to get correct dimensions.
        :return: Resolution
        """
        mm_per_inch = 25.4
        return (self._square_length_pix * mm_per_inch) / self._square_length_mm



    def json_save(self, filename):
        """ Save object parameters to json file
        :param filename: Filename of json file
        """
        params = {}
        self.dict_save(params)
        with open(filename, 'w') as file_handle:
            json.dump(params, file_handle, indent=4, sort_keys=True)



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        param_dict['squares'] = self._squares.tolist()
        param_dict['square_length_pix'] = self._square_length_pix
        param_dict['square_length_mm'] = self._square_length_mm
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
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        self._squares = np.asarray(param_dict['squares'], dtype=int)
        self._square_length_pix = param_dict['square_length_pix']
        self._square_length_mm = param_dict['square_length_mm']
        self._pose = Trafo3d()
        self._pose.dict_load(param_dict['pose'])



    def generate_image(self):
        """ Generates a 2D bitmap RGB image of the board
        :return: Image, shape (height, width, 3)
        """
        l = self._square_length_pix
        image = np.zeros((self._squares[1] * l, self._squares[0] * l, 3),
            dtype=np.uint8)
        for row in range(self._squares[1]):
            col_start = 1 if (row % 2) == 0 else 0
            for col in range(col_start, self._squares[0], 2):
                image[row*l:(row+1)*l, col*l:(col+1)*l, :] = 255
        return image



    def plot2d(self):
        """ Plots 2D image of board
        """
        image = self.generate_image()
        title = f'squares {self._squares}, shape {image.shape}, dpi {self.get_resolution_dpi():.0f}'
        image_show(image, title)



    def generate_mesh(self):
        """ Generates a 3D mesh object of the board
        width in X = self._squares[0] * self._square_length_mm
        height in Y = self._squares[1] * self._square_length_mm
        :return: Open3D mesh object
        """
        image = self.generate_image()
        mesh = mesh_generate_image(image, pixel_size=self.get_pixelsize_mm())
        mesh.transform(self._pose.get_homogeneous_matrix())
        return mesh



    def plot3d(self):
        """ Shows 3D image of board with a coordinate system
        """
        cs_size = np.min(self._squares) * self._square_length_mm
        cs = self.get_cs(cs_size)
        mesh = self.generate_mesh()
        o3d.visualization.draw_geometries([cs, mesh])



    def generate_screen(self):
        """ Generate a screen object with dimensions of board and its image
        :return: Screen object
        """
        image = self.generate_image()
        return Screen(self.get_size_mm(), image, self._pose)



    def max_num_points(self):
        """ Get maximum number of object and image points for calibration board
        """
        return np.prod(self._squares - 1)



    def get_object_points(self):
        """ Get all 3D object points of calibration board
        :return: 3D object points
        """
        obj_points = []
        for col in range(1, self._squares[1]):
            for row in reversed(range(1, self._squares[0])):
                obj_points.append([
                    col * self._square_length_mm, # X
                    row * self._square_length_mm, # Y
                    0.0                           # Z always zero
                ])
        return np.array(obj_points)



    def detect_obj_img_points(self, image):
        """ Detects object and image points in image
        :param image: RBG image of shape (height, width, 3)
        :param verbose: Show plot of detected corners and IDs
        :return: Lists of object points and image points
        """
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3 # RGB image
        assert image.dtype == np.uint8 # 8-bit
        # Detect image points
        success, corners = cv2.findChessboardCorners(image, self._squares - 1)
        if not success:
            raise Exception('Unable to detect checkboard coners.')
        img_points = np.array(corners).reshape((-1, 2))
        if img_points.shape[0] != self.max_num_points():
            raise Exception('Unable to detect all checkboard coners.')
        # Generate object points
        obj_points = self.get_object_points().astype(np.float32)
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.imshow(image)
            for i in range(img_points.shape[0]):
                ax.plot(img_points[i, 0], img_points[i, 1], '+r')
                ax.text(img_points[i, 0], img_points[i, 1], f'{i}', color='r')
            ax.set_axis_off()
            ax.set_title('img points')
            ax = fig.add_subplot(122)
            ax.imshow(self.generate_image())
            scale = self._square_length_pix / self._square_length_mm
            for i in range(obj_points.shape[0]):
                ax.plot(scale * obj_points[i, 0], scale * obj_points[i, 1], '+r')
                ax.text(scale * obj_points[i, 0], scale * obj_points[i, 1], f'{i}', color='r')
            ax.set_axis_off()
            ax.set_title('obj points')
            plt.show()
        return obj_points, img_points



    def detect_all_obj_img_points(self, images):
        """ Convenience function to detect all image points in a stack of images
        :param images: Stack of n images, shape (n, height, width, 3)
        :return list of object_points, list of image_points
        """
        obj_points = []
        img_points = []
        for i, image in enumerate(images):
            try:
                op, ip = self.detect_obj_img_points(image)
            except Exception as ex:
                raise type(ex)(str(ex) + f' (image {i})') from ex
            obj_points.append(op)
            img_points.append(ip)
        return obj_points, img_points



    def calibrate_intrinsics(self, images, flags=0):
        """ Calibrates intrinsics of a camera from a stack of images
        :param images: Stack of n images, shape (n, height, width, 3)
        :param flags: Calibration flags from OpenCV
        :return: Camera model, trafos from camera to each board, reprojection error
        """
        # Extract object and image points
        obj_points, img_points = self.detect_all_obj_img_points(images)
        # Calibrate camera
        image_shape = images.shape[1:3]
        reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, \
            image_shape, None, None, flags=flags)
        # Generate camera object and set intrincis
        cam = CameraModel()
        cam.set_chip_size((images.shape[2], images.shape[1]))
        cam.set_camera_matrix(camera_matrix)
        cam.set_distortion(dist_coeffs)
        return cam, reprojection_error