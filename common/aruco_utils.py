import sys
import os
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



class CharucoBoard:
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
        marker_length_mm=10.0, dict_type=aruco.DICT_6X6_250, ids=None):
        """ Constructor
        :param squares: Number of squares: height x width
        :param square_length_pix: Length of single square in pixels
        :param square_length_mm: Length of single square in millimeters
        :param marker_length_mm: Length of marker inside square in millimeters
        :param dict_type: Dictionary type
        :param ids: List of IDs for the aruco markers on the white chessboard squares
        """
        self._squares = np.asarray(squares)
        self._square_length_pix = square_length_pix
        self._square_length_mm = square_length_mm
        self._marker_length_mm = marker_length_mm
        self._dict_type = dict_type
        self._ids = ids



    def __str__(self):
        """ Get readable string representation of object
        :return: String representation of object
        """
        param_dict = {}
        self.dict_save(param_dict)
        return str(param_dict)



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
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
        """ Get size of a single pixel in millimeter
        :return: Size as scalar
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
        return mesh_generate_image(image, pixel_size=self.get_pixelsize_mm())



    def plot3d(self):
        """ Shows 3D image of board with a coordinate system
        """
        cs_size = np.min(self._squares) * self._square_length_mm
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cs_size)
        mesh = self.generate_mesh()
        o3d.visualization.draw_geometries([cs, mesh])



    def generate_screen(self):
        """ Generate a screen object with dimensions of board and its image
        :return: Screen object
        """
        image = self.generate_image()
        return Screen(self.get_size_mm(), image)



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
        return obj_points, img_points



    def detect_obj_img_points(self, images):
        """
        Extracts object points (3D) from board, detects image points (2D)
        in each image of a stack of images and matches object and image
        points to each other
        :param images: Stack of n images, shape (n, height, width, 3)
        :return: object points, image points, corners and ids
        """
        assert isinstance(images, np.ndarray)
        assert images.ndim == 4
        assert images.shape[3] == 3 # RGB image
        assert images.dtype == np.uint8 # 8-bit
        board = self._generate_board()
        # TODO add CharucoParameters to activate refinement of markers or
        # to provide camera matrix and distortion parameters to detection;
        # causes segfault in OpenCV now
        # https://github.com/opencv/opencv/issues/23440
        detector = aruco.CharucoDetector(board)
        all_obj_points = []
        all_img_points = []
        all_corners = []
        all_ids = []
        for i in range(images.shape[0]):
            # Detection of markers and corners
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                detector.detectBoard(images[i, :, :, :])
            if charuco_corners is None:
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

            if obj_points.shape[0] < 4:
                raise Exception('Not enough matching object-/imagepoint pairs.')
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        return all_obj_points, all_img_points, all_corners, all_ids



    def _annotate_image(self, image, corners, ids, trafo, camera_matrix, dist_coeffs):
        annotated_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        aruco.drawDetectedCornersCharuco(annotated_image, corners,
            ids, (255, 0, 255))
        rvec = trafo.get_rotation_rodrigues()
        tvec = trafo.get_translation()
        cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, \
            rvec, tvec, self._square_length_mm)
        return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)



    def calibrate(self, images, flags=0, verbose=False):
        """ Calibrates a camera from a stack of images
        :param images: Stack of n images, shape (n, height, width, 3)
        :param flags: Calibration flags from OpenCV
        :param verbose: Show plots of markers and coordinate system that have been found
        :return: Camera model, trafos from camera to each board, reprojection error
        """
        # Extract object and image points
        obj_points, img_points, corners, ids = \
            self.detect_obj_img_points(images)
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
        # If requested, visualize result
        if verbose:
            annotated_images = []
            for i in range(images.shape[0]):
                annotated_image = self._annotate_image(images[i], corners[i], ids[i],
                    cam_to_boards[i], camera_matrix, dist_coeffs)
                annotated_images.append(annotated_image)
            annotated_images = np.asarray(annotated_images)
            image_show_multiple(annotated_images)
        return cam, cam_to_boards, reprojection_error



    def estimate_pose(self, image, cam, verbose=False):
        """ Estimates the pose of a Charuco board in an image
        Estimates the pose of a calibrated camera relative to a Charuco board
        using a image taken with the camera of said board.
        :param image: Image, shape (height, width, 3)
        :param cam: CameraModel object with known model parameters
        :param verbose: Show plot of markers and coordinate system that have been found
        :return: Trafo from camera to board
        """
        # Extract object and image points
        images = np.array((image, ))
        obj_points, img_points, corners, ids = \
            self.detect_obj_img_points(images)
        # Find an object pose from 3D-2D point correspondences
        camera_matrix = cam.get_camera_matrix()
        dist_coeffs = cam.get_distortion()
        retval, rvec, tvec = cv2.solvePnP(obj_points[0], img_points[0], \
            camera_matrix, dist_coeffs)
        # Convert into Trafo3d object
        cam_to_board = Trafo3d(rodr=rvec, t=tvec)
        # If requested, visualize result
        if verbose:
            annotated_image = self._annotate_image(image, corners[0], ids[0],
                cam_to_board, camera_matrix, dist_coeffs)
            image_show(annotated_image)
        return cam_to_board



class MultiMarker:

    def __init__(self, length_pix=80, length_mm=20.0, \
            dict_type=aruco.DICT_6X6_250, pose=Trafo3d()):
        self._length_pix = length_pix
        self._length_mm = length_mm
        self._dict_type = dict_type
        self._pose = pose # world_to_center
        self._markers = {}



    def __str__(self):
        """ Get readable string representation of object
        :return: String representation of object
        """
        param_dict = {}
        self.dict_save(param_dict)
        return str(param_dict)



    def add_marker(self, id, center_to_marker):
        if id in self._markers:
            raise Exception('Failure adding duplicate ID.')
        self._markers[id] = center_to_marker



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        param_dict['length_pix'] = self._length_pix
        param_dict['length_mm'] = self._length_mm
        param_dict['dict_type'] = self._dict_type
        param_dict['pose'] = {}
        self._pose.dict_save(param_dict['pose'])
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
        self._length_pix = param_dict['length_pix']
        self._length_mm = param_dict['length_mm']
        self._dict_type = param_dict['dict_type']
        self._pose = Trafo3d()
        self._pose.dict_load(param_dict['pose'])
        self._markers = {}
        for marker in param_dict['markers']:
            id = marker['id']
            trafo = Trafo3d()
            trafo.dict_load(marker['pose'])
            self._markers[id] = trafo



    def get_pixelsize_mm(self):
        return self._length_mm / self._length_pix



    def get_resolution_dpi(self):
        mm_per_inch = 25.4
        return (self._length_pix * mm_per_inch) / self._length_mm



    def generate_images(self):
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        images = np.zeros((len(self._markers), self._length_pix, self._length_pix, 3),
            dtype=np.uint8)
        for i, (id, trafo) in enumerate(self._markers.items()):
            image = aruco_dict.generateImageMarker(id, self._length_pix)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            images[i, :, :, :] = image
        return images



    def plot2d(self):
        images = self.generate_images()
        titles = [ f'image {i}: id #{id}' for i, id in enumerate(self._markers.keys()) ]
        image_show_multiple(images, titles, single_window=True)



    def generate_meshes(self):
        images = self.generate_images()
        meshes = []
        for trafo, image in zip(self._markers.values(), images):
            mesh = mesh_generate_image(image, pixel_size=self.get_pixelsize_mm())
            T = self._pose * trafo
            mesh.transform(T.get_homogeneous_matrix())
            meshes.append(mesh)
        return meshes



    def generate_screens(self):
        images = self.generate_images()
        dimensions = (self._length_mm, self._length_mm)
        screens = []
        for i, trafo in enumerate(self._markers.values()):
            screen = Screen(dimensions, images[i], self._pose * trafo)
            screens.append(screen)
        return screens



    def plot3d(self):
        cs_size = self._length_mm
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cs_size)
        cs.transform(self._pose.get_homogeneous_matrix())
        objects = [ cs ]
        screens = self.generate_screens()
        for screen in screens:
            objects.append(screen.get_cs(size=cs_size))
            objects.append(screen.get_mesh())
        o3d.visualization.draw_geometries(objects)



    def _get_object_points(self, id):
        """ Get object points for give ID in center coordinate system
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
        obj_points = []
        img_points = []
        for corner, id in zip(corners, ids):
            # Unknown ID may come from same aruco dict
            # but from different MultiMarker object, so ignore
            if id[0] not in self._markers:
                continue
            obj_points.append(self._get_object_points(id[0]))
            img_points.append(corner[0, :, :])
        obj_points = np.array(obj_points).reshape((-1, 3))
        img_points = np.array(img_points).reshape((-1, 2))
        return obj_points, img_points



    def _detect_obj_img_points(self, image, verbose=False):
        assert isinstance(image, np.ndarray)
        assert image.ndim == 3
        assert image.shape[2] == 3 # RGB image
        assert image.dtype == np.uint8 # 8-bit
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        detector = aruco.ArucoDetector(aruco_dict)
        # TODO add ArucoParameters to activate refinement of markers or
        # to provide camera matrix and distortion parameters to detection;
        # causes segfault in OpenCV now
        # https://github.com/opencv/opencv/issues/23440
        corners, ids, rejectedImgPoints = detector.detectMarkers(image)
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(image)
            margin = 6
            for corner, id in zip(corners, ids):
                for i in range(4):
                    ax.plot(corner[0, i, 0], corner[0, i, 1], 'or')
                    ax.text(corner[0, i, 0] + margin, corner[0, i, 1] - margin,
                        f'{i}', color='r', fontsize=16)
                xy = np.mean(corner[0], axis=0)
                ax.text(*xy, f'{id[0]}', color='g', fontsize=20)
        return self._match_aruco_corners(corners, ids)



    def _objfun(x, obj_points, img_points, cams):
        world_to_center = Trafo3d(t=x[:3], rodr=x[3:])
        residuals = []
        for i in range(len(cams)):
            op = world_to_center * obj_points[i]
            ip = cams[i].scene_to_chip(op)
            residuals.append(ip[:, 0:2] - img_points[i])
        return np.vstack(residuals).ravel()



    def estimate_pose(self, cams, images):
        assert len(cams) == len(images)
        # Detect object and image points
        obj_points = []
        img_points = []
        for image in images:
            op, ip = self._detect_obj_img_points(image)
            obj_points.append(op)
            img_points.append(ip)
        # TODO: Find good start values
        x0 = np.array((-400.0, 100.0, 0.0, 0.0, 0.0, 0.0))
        res = least_squares(MultiMarker._objfun, x0, args=(obj_points, img_points, cams))
        if not res.success:
            raise Exception(f'Numerical optimization failed: {res.message}')
        world_to_center = Trafo3d(t=res.x[:3], rodr=res.x[3:])
        return world_to_center

