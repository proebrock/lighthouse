import sys
import os
import copy
from abc import ABC, abstractmethod
import cv2
import cv2.aruco as aruco
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_show, image_show_multiple
from common.mesh_utils import mesh_generate_image
from camsimlib.screen import Screen
from camsimlib.camera_model import CameraModel



class MultiMarker(ABC):

    def __init__(self, pose):
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
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        cs.transform(self._pose.get_homogeneous_matrix())
        return cs



    @abstractmethod
    def dict_save(self, param_dict):
        param_dict['pose'] = {}
        self._pose.dict_save(param_dict['pose'])



    @abstractmethod
    def dict_load(self, param_dict):
        self._pose = Trafo3d()
        self._pose.dict_load(param_dict['pose'])



    @abstractmethod
    def detect_obj_img_points(self, image):
        pass



    def detect_all_obj_img_points(self, images):
        obj_points = []
        img_points = []
        for image in images:
            op, ip = self.detect_obj_img_points(image)
            obj_points.append(op)
            img_points.append(ip)
        return obj_points, img_points



    @staticmethod
    def _solve_pnp(cam, obj_points, img_points):
        assert obj_points.shape[0] == img_points.shape[0]
        if obj_points.shape[0] < 4:
            return None
        camera_matrix = cam.get_camera_matrix()
        dist_coeffs = cam.get_distortion()
        retval, rvec, tvec = cv2.solvePnP(obj_points, img_points, \
            camera_matrix, dist_coeffs)
        # Convert into Trafo3d object
        cam_to_object = Trafo3d(rodr=rvec, t=tvec)
        return cam_to_object



    @staticmethod
    def _objfun_pose(x, obj_points, img_points, cams):
        """ Objective function for optimization used in pose estimation
        :param x: Decision variable: Trafo from world to center as translation and rodrigues vector
        :param obj_points: List of marker object points for each camera
        :param img_points: List of marker image points for each camera
        :param cams: List of CameraModel objects
        :return: Residuals for all image point differences in x/y
        """
        world_to_center = Trafo3d(t=x[:3], rodr=x[3:])
        residuals = []
        for i, cam in enumerate(cams):
            op = world_to_center * obj_points[i]
            ip = cam.scene_to_chip(op)
            residuals.append(ip[:, 0:2] - img_points[i])
        return np.vstack(residuals).ravel()



    def estimate_pose(self, cams, images):
        """ Estimate the pose of the MultiMarker object
        by using a list of calibrated cameras and a list of single images from
        each of the cameras; estimates trafo from world to center of
        the MultiMarker object.
        :param cams: List of CameraModel objects
        :param images: List of images, shape (height, width, 3),
            height and width fits camera resolutions
        :return: Estimated trafo of type Trafo3d
        """
        # Check consistency of inputs
        assert len(cams) == len(images)
        for image, cam in zip(images, cams):
            sh = image.shape
            cs = cam.get_chip_size()
            assert sh[0] == cs[1]
            assert sh[1] == cs[0]
            assert sh[2] == 3 # RGB
        obj_points, img_points = self.detect_all_obj_img_points(images)
        # Find start value: SolvePnP with single camera
        img_init_index = 0
        cam_to_center = MultiMarker._solve_pnp( \
            cams[img_init_index],
            obj_points[img_init_index],
            img_points[img_init_index])
        world_to_cam = cams[img_init_index].get_pose()
        world_to_center0 = world_to_cam * cam_to_center
        x0 = np.concatenate((
            world_to_center0.get_translation(),
            world_to_center0.get_rotation_rodrigues(),
        ))
        # Run optimization
        res = least_squares(MultiAruco._objfun_pose, x0,
            args=(obj_points, img_points, cams))
        if not res.success:
            raise Exception(f'Numerical optimization failed: {res.message}')
        world_to_center = Trafo3d(t=res.x[:3], rodr=res.x[3:])
        residuals = MultiAruco._objfun_pose(res.x, obj_points, img_points, cams)
        residuals_rms = np.sqrt(np.mean(np.square(residuals)))
        if False:
            # Assess situation BEFORE optimization
            residuals_x0 = MultiAruco._objfun_pose(x0, obj_points, img_points, cams)
            residuals_rms_x0 = np.sqrt(np.mean(np.square(residuals_x0)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(residuals_x0, label='before opt')
            ax.plot(residuals, label='after opt')
            ax.grid()
            ax.set_title(f'Residuals RMS: {residuals_rms_x0:.2f} -> {residuals_rms:.2f}')
            ax.legend()
            plt.show()
        return world_to_center, residuals_rms



    @staticmethod
    def _params_to_x(world_to_cams, world_to_markers):
        """ Converts the transformation to be optimized into decision variable vector
        :param world_to_cams: List of Trafo3d: world to cameras
        :param world_to_markers: List of Trafo3d: world to marker centers
        :return: Decision variable vector
        """
        x = []
        cam_index_world = 0 # Index of camera that is the world coordinate system
        for i in range(len(world_to_cams)):
            if i == cam_index_world:
                continue
            x.extend(world_to_cams[i].get_translation())
            x.extend(world_to_cams[i].get_rotation_rodrigues())
        for i in range(len(world_to_markers)):
            x.extend(world_to_markers[i].get_translation())
            x.extend(world_to_markers[i].get_rotation_rodrigues())
        return np.asarray(x)



    @staticmethod
    def _x_to_params(x, num_cams):
        """ Converts a decision variable vector to the transformations represented by it
        :param x: Decision variable vector
        :param num_cams: Number of cameras
        :return: Lists of Trafo3d: (world_to_cams, world_to_markers)
        """
        trafos = []
        for i in range(0, x.size, 6):
            trafos.append(Trafo3d(t=x[i:i+3], rodr=x[i+3:i+6]))
        world_to_cams = trafos[0:num_cams-1]
        cam_index_world = 0
        world_to_cams.insert(cam_index_world, Trafo3d())
        world_to_markers = trafos[num_cams-1:]
        return world_to_cams, world_to_markers



    @staticmethod
    def _objfun_excalib(x, obj_points, img_points, cams):
        """ Objective function for optimization used in extrinsic camera calibration
        :param x: Decision variable (various transformation, see _params_to_x)
        :param obj_points: List of marker object points for each camera and each image
        :param img_points: List of marker image points for each camera and each image
        :param cams: List of CameraModel objects
        :return: Residuals for all image point differences in x/y
        """
        num_cams = len(cams)
        num_imgs = len(obj_points[0])
        world_to_cams, world_to_markers = MultiMarker._x_to_params(x, num_cams)
        residuals = []
        for i in range(num_cams):
            cams[i].set_pose(world_to_cams[i])
            for j in range(num_imgs):
                op = world_to_markers[j] * obj_points[i][j]
                ip = cams[i].scene_to_chip(op)
                residuals.append(ip[:, 0:2] - img_points[i][j])
        return np.vstack(residuals).ravel()



    def calibrate_extrinsics(self, cams, image_stacks):
        """ Estimates the extrinsic calibration of multiple
        calibrated cameras (intrinsics known)
        :param cams: List of CameraModel objects
        :param image_stacks: List of image stacks, each image stack must have same
            number of images and shape of i-th image stack in the list must fit
            camera resolution of i-th camera
        :return: List of
        """
        # Check consistency of inputs
        assert len(cams) == len(image_stacks)
        num_cams = len(cams)
        num_imgs = None
        for images, cam in zip(image_stacks, cams):
            sh = images.shape
            cs = cam.get_chip_size()
            if num_imgs is None:
                num_imgs = sh[0]
            else:
                assert num_imgs == sh[0]
            assert sh[1] == cs[1]
            assert sh[2] == cs[0]
            assert sh[3] == 3 # RGB
        # Extract object and image points
        obj_points = []
        img_points = []
        for i in range(num_cams):
            op, ip = self.detect_all_obj_img_points(image_stacks[i])
            obj_points.append(op)
            img_points.append(ip)
        # Make initial estimates for camera positions
        img_init_index = 0
        cams_to_marker = []
        for i in range(num_cams):
            cam_to_marker = MultiMarker._solve_pnp(cams[i], \
                obj_points[i][img_init_index],
                img_points[i][img_init_index])
            if cam_to_marker is None:
                raise Exception(f'Unable to get initial estimate of camera pose {i}')
            cams_to_marker.append(cam_to_marker)
        # Transform everything relative to cam0 = world
        world_to_cams = []
        for i in range(num_cams):
            world_to_cams.append(cams_to_marker[0] * cams_to_marker[i].inverse())
        # Make initial estimates for marker object positions
        cam_init_index = 0
        cam_to_markers = []
        for i in range(num_imgs):
            cam_to_marker = MultiMarker._solve_pnp(cams[cam_init_index], \
                obj_points[cam_init_index][i],
                img_points[cam_init_index][i])
            if cam_to_marker is None:
                raise Exception(f'Unable to get initial estimate of marker pose {i}')
            cam_to_markers.append(cam_to_marker)
        # Transform everything relative to cam0 = world
        world_to_markers = []
        for i in range(num_imgs):
            world_to_markers.append(world_to_cams[cam_init_index] * cam_to_markers[i])
        # Run optimization
        x0 = MultiMarker._params_to_x(world_to_cams, world_to_markers)
        res = least_squares(MultiAruco._objfun_excalib, x0,
            args=(obj_points, img_points, cams))
        if not res.success:
            raise Exception(f'Numerical optimization failed: {res.message}')
        residuals = MultiAruco._objfun_excalib(res.x, obj_points, img_points, cams)
        residuals_rms = np.sqrt(np.mean(np.square(residuals)))
        world_to_cams_final, world_to_markers_final = MultiMarker._x_to_params(res.x, num_cams)
        # Write extrinsics to cameras
        for cam, pose in zip(cams, world_to_cams_final):
            cam.set_pose(pose)
        if False:
            # Assess situation BEFORE optimization
            residuals_x0 = MultiAruco._objfun_excalib(x0, obj_points, img_points, cams)
            residuals_rms_x0 = np.sqrt(np.mean(np.square(residuals_x0)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(residuals_x0, label='before opt')
            ax.plot(residuals, label='after opt')
            ax.grid()
            ax.set_title(f'Residuals RMS: {residuals_rms_x0:.2f} -> {residuals_rms:.2f}')
            ax.legend()
            plt.show()
        return world_to_cams_final, world_to_markers_final, residuals_rms



class CharucoBoard(MultiMarker):
    """ Representation of a Charuco board usable for calibration and pose estimation.

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

    def __init__(self, squares=(5, 7), square_length_pix=80, square_length_mm=20.0,
        marker_length_mm=10.0, dict_type=aruco.DICT_6X6_250, ids=None, pose=Trafo3d()):
        """ Constructor
        :param squares: Number of squares: height x width
        :param square_length_pix: Length of single square in pixels
        :param square_length_mm: Length of single square in millimeters
        :param marker_length_mm: Length of marker inside square in millimeters
        :param dict_type: Aruco dictionary type
        :param ids: List of IDs for the aruco markers on the white chessboard squares
        :param pose: Transformation from world to CharucoBoard
        """
        super(CharucoBoard, self).__init__(pose)
        self._squares = np.asarray(squares)
        self._square_length_pix = square_length_pix
        self._square_length_mm = square_length_mm
        self._marker_length_mm = marker_length_mm
        self._dict_type = dict_type
        self._ids = ids



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        super(CharucoBoard, self).dict_save(param_dict)
        param_dict['squares'] = self._squares.tolist()
        param_dict['square_length_pix'] = self._square_length_pix
        param_dict['square_length_mm'] = self._square_length_mm
        param_dict['marker_length_mm'] = self._marker_length_mm
        param_dict['dict_type'] = self._dict_type
        if self._ids is None:
            param_dict['ids'] = None
        else:
            param_dict['ids'] = self._ids.tolist()



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        super(CharucoBoard, self).dict_load(param_dict)
        self._squares = np.asarray(param_dict['squares'], dtype=int)
        self._square_length_pix = param_dict['square_length_pix']
        self._square_length_mm = param_dict['square_length_mm']
        self._marker_length_mm = param_dict['marker_length_mm']
        self._dict_type = param_dict['dict_type']
        ids = param_dict['ids']
        if ids is None:
            self._ids = None
        else:
            self._ids = np.asarray(ids, dtype=int)



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



    def _generate_board(self):
        """ Generates an object of type cv2.aruco.CharucoBoard
        Used for generating board representations and for detections
        :return: Board
        """
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        board = aruco.CharucoBoard(self._squares, self._square_length_mm,
            self._marker_length_mm, aruco_dict, self._ids)
        return board



    def generate_image(self):
        """ Generates a 2D bitmap RGB image of the board
        width = self._squares[0] * self._square_length_pix
        height = self._squares[1] * self._square_length_pix
        :return: Image, shape (height, width, 3)
        """
        board = self._generate_board()
        image = board.generateImage(self.get_size_pix())
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)



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



    @staticmethod
    def _plot_corners_ids(charuco_corners, charuco_ids, marker_corners, marker_ids, image):
        """ Plots charuco corners/ids and marker corners/ids in separate subplots
        :param charuco_corners: Charuco corners
        :param charuco_ids: Charuco ids
        :param marker_corners: Marker corners
        :param marker_ids: Marker ids
        :param image: Image used as background for plotting
        """
        cc = np.asarray(charuco_corners).reshape((-1, 2))
        ci = np.asarray(charuco_ids).ravel()
        assert cc.shape[0] == ci.size
        mc = np.asarray(marker_corners).reshape((-1, 4, 2))
        mi = np.asarray(marker_ids).ravel()
        assert mc.shape[0] == mi.size
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # Image
        ax.imshow(image)
        # Charuco corners
        margin = 6
        for i in range(cc.shape[0]):
            ax.plot(cc[i, 0], cc[i, 1], 'og')
            ax.text(cc[i, 0] + margin, cc[i, 1] - margin, f'{ci[i]}', color='g')
        # Marker corners
        for i in range(mc.shape[0]):
            for j in range(mc.shape[1]):
                ax.plot(mc[i, j, 0], mc[i, j, 1], 'or')
                ax.text(mc[i, j, 0], mc[i, j, 1], f'{j}')
            xy = np.mean(mc[i, :, :], axis=0)
            ax.text(xy[0], xy[1], f'{mi[i]}', color='r')
        legend_elements = [ \
            Line2D([0], [0], marker='o', color='w', label='charuco corners',
                markerfacecolor='g'),
            Line2D([0], [0], marker='o', color='w', label='marker corners',
                markerfacecolor='r'),
        ]
        ax.legend(handles=legend_elements)



    @staticmethod
    def _plot_correspondences(obj_points, img_points, image, max_num_corr = 16):
        """ Plots matching object points and image points in separate subplots
        and connect matching points with lines; requires that the number of
        object points and image points are the same and the i-th object point
        corresponds to the i-th image point.
        Max number of correspondences lines plotted is limited to keep image
        still readable.
        :param obj_points: Object points, shape (n, 3), zero Z coordinates
        :param img_points: Image points, shape (n, 2)
        :param image: Image used as background for image point plotting
        :param max_num_corr: Max number of correspondences plotted
        """
        # Check for consistency
        assert obj_points.shape[0] == img_points.shape[0]
        assert obj_points.shape[-1] == 3
        assert img_points.shape[-1] == 2
        n = obj_points.shape[0]
        objp = obj_points.reshape((n, 3))
        # The board is a plane, so we expect all Z values to be zero
        assert np.all(np.isclose(objp[:, 2], 0.0))
        objp = objp[:, 0:2] # Omit Z values
        imgp = img_points.reshape((n, 2))
        fig = plt.figure()
        # Object points
        axo = fig.add_subplot(121)
        axo.plot(objp[:, 0], objp[:, 1], 'xb')
        axo.set_xlabel('x (mm)')
        axo.set_ylabel('y (mm)')
        axo.set_title('obj_points')
        # Image points
        axi = fig.add_subplot(122)
        axi.imshow(image)
        axi.plot(imgp[:, 0], imgp[:, 1], 'xb')
        axi.set_title('img_points')
        # Correspondences
        indices = np.random.choice(n, np.min(n, max_num_corr), replace=False)
        for i in indices:
            p = axo.plot(objp[i, 0], objp[i, 1], 'o')
            color = p[0].get_color()
            axi.plot(imgp[i, 0], imgp[i, 1], 'o', color=color)
            con = ConnectionPatch(xyA=objp[i, :], xyB=imgp[i, :],
                coordsA="data", coordsB="data", axesA=axo, axesB=axi, color=color)
            axi.add_artist(con)
        plt.show()



    @staticmethod
    def _match_charuco_corners(board, charuco_corners, charuco_ids):
        """ Matches a set of charuco corners and ids to the chessboard corners
        of a charuco board
        :param board: Charuco board
        :param charuco_corners: Charuco corners
        :param charuco_ids: Charuco ids
        :return: Matching object and image points
        """
        assert charuco_corners.shape[0] == charuco_ids.shape[0]
        assert charuco_corners.shape[-1] == 2 # 2 points on image
        cbc = board.getChessboardCorners()
        assert np.all(charuco_ids[:, 0] < cbc.shape[0])
        obj_points = cbc[charuco_ids[:, 0]].reshape((-1, 1, 3))
        img_points = charuco_corners.copy()
        assert obj_points.shape[0] == img_points.shape[0]
        obj_points = np.array(obj_points).reshape((-1, 3))
        img_points = np.array(img_points).reshape((-1, 2))
        return obj_points, img_points



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
        board = self._generate_board()
        # TODO add CharucoParameters to activate refinement of markers or
        # to provide camera matrix and distortion parameters to detection;
        # causes segfault in OpenCV now
        # https://github.com/opencv/opencv/issues/23440
        detector = aruco.CharucoDetector(board)
        # Detection of markers and corners
        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            detector.detectBoard(image)
        if charuco_corners is None or charuco_ids is None:
            raise Exception('No charuco corners detected.')
        #self._plot_corners_ids(charuco_corners, charuco_ids, marker_corners, marker_ids, images[i])

        # TODO: Official example show the usage of charuco_corners/charuco_ids
        # instead of marker_corners/marker_ids for calibration and detection;
        # but this seems to lead to terrible calibration results for unknown
        # reason. This must be investigated.
        # Solving this is a pre-condition for using specific IDs from self._ids
        #obj_points, img_points = board.matchImagePoints( \
        #    marker_corners, marker_ids)

        # Matching of corners in order to get object and image point pairs
        obj_points, img_points = self._match_charuco_corners(board, charuco_corners, charuco_ids)
        #self._plot_correspondences(obj_points, img_points, images[i, :, :, :])
        return obj_points, img_points



    def _annotate_image(self, image, corners, ids, trafo, cam):
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        aruco.drawDetectedCornersCharuco(annotated_image, corners,
            ids, (255, 0, 255))
        camera_matrix = cam.get_camera_matrix()
        dist_coeffs = cam.get_distortion()
        rvec = trafo.get_rotation_rodrigues()
        tvec = trafo.get_translation()
        cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, \
            rvec, tvec, self._square_length_mm)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)



    def calibrate_intrinsics(self, images, flags=0):
        """ Calibrates intrinsics of a camera from a stack of images
        :param images: Stack of n images, shape (n, height, width, 3)
        :param flags: Calibration flags from OpenCV
        :param verbose: Show plots of markers and coordinate system that have been found
        :return: Camera model, trafos from camera to each board, reprojection error
        """
        # Extract object and image points
        obj_points, img_points = self.detect_all_obj_img_points(images)
        # Calibrate camera
        image_shape = images.shape[1:3]
        reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, \
            image_shape, None, None, flags=flags)
        # Set intrincis
        cam = CameraModel()
        cam.set_chip_size((images.shape[2], images.shape[1]))
        cam.set_camera_matrix(camera_matrix)
        cam.set_distortion(dist_coeffs)
        # Set extrinsics
        cam_to_boards = []
        for rvec, tvec in zip(rvecs, tvecs):
            cam_to_boards.append(Trafo3d(rodr=rvec, t=tvec))
        return cam, cam_to_boards, reprojection_error



class MultiAruco(MultiMarker):
    """ Representation of an object carrying one or multiple aruco markers
    with fixed transformations between each other.

    Intended use is pose estimation of the MultiAruco object and extrinsics
    calibration of multi camera setups.

    The object has a coordinate system called "center". Each markers position
    is relative to that center CS, self._markers[id] contains the transformation
    center to marker. The object keeps a pose in self._pose which is the
    transformation from world to center.

    A single marker is square with given side length. The coordinate system
    is as shown. The indices denote the order in which the detector returns
    the image points of the marker in an image.

                Z        X
                    X ------->
                    |
                Y   |  0 .----------. 1
                    |    |          |
                    V    |  Aruco   |
                         |  Marker  |
                         |          |
                       3 .----------. 2

    All markers in MultiAruco have the same size and are based on the same
    aruco dictionary.
    """

    def __init__(self, length_pix=80, length_mm=20.0, \
            dict_type=aruco.DICT_6X6_250, pose=Trafo3d()):
        """ Constructor
        :param length_pix: Length of marker in pixels
        :param length_mm: Length of marker in millimeters
        :param dict_type: Aruco dictionary type
        :param pose: Transformation from world to center of MultiAruco object
        """
        super(MultiAruco, self).__init__(pose)
        self._length_pix = length_pix
        self._length_mm = length_mm
        self._dict_type = dict_type
        self._markers = {}



    def add_marker(self, id, center_to_marker):
        """ Add a new marker to the MultiAruco object
        :param id: Identifier of marker from aruco dictionary, must be unique
        :param center_to_marker: Location of the marker relative to center CS
        """
        if id in self._markers:
            raise Exception('Failure adding duplicate ID.')
        self._markers[id] = center_to_marker



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        super(MultiAruco, self).dict_save(param_dict)
        param_dict['length_pix'] = self._length_pix
        param_dict['length_mm'] = self._length_mm
        param_dict['dict_type'] = self._dict_type
        param_dict['markers'] = []
        for id, trafo in self._markers.items():
            marker = {}
            marker['id'] = id
            marker['pose'] = {}
            trafo.dict_save(marker['pose'])
            param_dict['markers'].append(marker)



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        super(MultiAruco, self).dict_load(param_dict)
        self._length_pix = param_dict['length_pix']
        self._length_mm = param_dict['length_mm']
        self._dict_type = param_dict['dict_type']
        self._markers = {}
        for marker in param_dict['markers']:
            id = marker['id']
            trafo = Trafo3d()
            trafo.dict_load(marker['pose'])
            self._markers[id] = trafo



    def get_pixelsize_mm(self):
        """ Get size of a single pixel in millimeters
        :return: Size as a scalar
        """
        return self._length_mm / self._length_pix



    def get_resolution_dpi(self):
        """ Get resolution of marker in DPI (dots per inch)
        Use this resolution to print the marker on paper to get correct dimensions.
        :return: Resolution
        """
        mm_per_inch = 25.4
        return (self._length_pix * mm_per_inch) / self._length_mm



    def generate_images(self):
        """ Generates a stack of 2D bitmap RGB images of all markers
        :return: Image stack, shape (num_markers, _length_pix, _length_pix, 3)
        """
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        images = np.zeros((len(self._markers), self._length_pix, self._length_pix, 3),
            dtype=np.uint8)
        for i, (id, trafo) in enumerate(self._markers.items()):
            image = aruco_dict.generateImageMarker(id, self._length_pix)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            images[i, :, :, :] = image
        return images



    def plot2d(self):
        """ Plots 2D images of the MultiAruco object
        """
        images = self.generate_images()
        titles = [ f'image {i}: id #{id}' for i, id in enumerate(self._markers.keys()) ]
        image_show_multiple(images, titles, single_window=True)



    def generate_mesh(self):
        """ Generates a 3D mesh object of the board
        :return: Open3D mesh object
        """
        images = self.generate_images()
        meshes = o3d.geometry.TriangleMesh()
        for trafo, image in zip(self._markers.values(), images):
            mesh = mesh_generate_image(image, pixel_size=self.get_pixelsize_mm())
            T = self._pose * trafo
            mesh.transform(T.get_homogeneous_matrix())
            meshes += mesh
        return meshes



    def generate_screens(self):
        """ Generate a list of screen objects, one for each marker, all properly
        aligned with their individual poses and self._pose
        :return: List of screen objects
        """
        images = self.generate_images()
        dimensions = (self._length_mm, self._length_mm)
        screens = []
        for i, trafo in enumerate(self._markers.values()):
            screen = Screen(dimensions, images[i], self._pose * trafo)
            screens.append(screen)
        return screens



    def plot3d(self):
        """ Shows 3D image of all markers of the MultiAruco object with CSs
        """
        cs_size = self._length_mm
        cs = self.get_cs(cs_size)
        objects = [ cs ]
        screens = self.generate_screens()
        for screen in screens:
            objects.append(screen.get_cs(size=cs_size))
            objects.append(screen.get_mesh())
        o3d.visualization.draw_geometries(objects)



    def _get_object_points(self, id):
        """ Get object points for a given ID in center coordinate system
        Object points are the four corner points of the marker described
        in the center coordinate system of the MultiAruco object.
        :param id: Identifier of marker in MultiAruco object
        :return: Object points, shape (4, 3)
        """
        if id not in self._markers:
            raise Exception(f'Unknown id {id}')
        obj_points = self._length_mm * np.array((
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ))
        center_to_marker = self._markers[id]
        return center_to_marker * obj_points



    def _match_aruco_corners(self, corners, ids):
        """ Matches a list of detected corners and IDs and
        generates object and image points.
        If the list contains an id that is not part of the
        MultiAruco object, it is ignored.
        :param corners: Corners as extracted by aruco.ArucoDetector
        :param ids: IDs as extracted by aruco.ArucoDetector
        :return: Lists of object points and image points
        """
        obj_points = []
        img_points = []
        for corner, id in zip(corners, ids):
            # Unknown ID may come from same aruco dict
            # but from different MultiAruco object, so ignore
            if id[0] not in self._markers:
                continue
            obj_points.append(self._get_object_points(id[0]))
            img_points.append(corner[0, :, :])
        obj_points = np.array(obj_points).reshape((-1, 3))
        img_points = np.array(img_points).reshape((-1, 2))
        return obj_points, img_points



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
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        # TODO add ArucoParameters to activate refinement of markers or
        # to provide camera matrix and distortion parameters to detection;
        # causes segfault in OpenCV now
        # https://github.com/opencv/opencv/issues/23440
        detector = aruco.ArucoDetector(aruco_dict)
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        if corners is None or ids is None:
            obj_points = np.zeros((0, 3))
            img_points = np.zeros((0, 2))
            return obj_points, img_points
        return self._match_aruco_corners(corners, ids)
