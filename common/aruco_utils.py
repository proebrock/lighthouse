import sys
import os
import cv2
import cv2.aruco as aruco
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.mesh_utils import mesh_generate_image
from camsimlib.screen import Screen



class CharucoBoard:

    def __init__(self, squares, square_length_pix, square_length_mm, marker_length_mm,
        dict_type=aruco.DICT_6X6_250):
        """ Constructor
        :param squares: Number of squares: height x width
        :param square_length_pix: Length of single square in pixels
        :param square_length_mm: Length of single square in millimeters
        :param marker_length_mm: Length of marker inside square in millimeters
        :param dict_type: Dictionary type
        """
        self._squares = np.asarray(squares)
        self._square_length_pix = square_length_pix
        self._square_length_mm = square_length_mm
        self._marker_length_mm = marker_length_mm
        self._dict_type = dict_type
        # From OpenCV version 4.6 the location of the coordinate system changed:
        # it moved from one corner to the other and the Z-Axis was facing inwards;
        # this corrective transformation compensates for it.
        dy = self._squares[1] * self._square_length_mm
        self._T_CORR = Trafo3d(t=(0, dy, 0), rpy=(np.pi, 0, 0))



    def __str__(self):
        param_dict = {}
        self.dict_save(param_dict)
        return str(param_dict)



    def dict_save(self, param_dict):
        param_dict['squares'] = self._squares.tolist()
        param_dict['square_length_pix'] = self._square_length_pix
        param_dict['square_length_mm'] = self._square_length_mm
        param_dict['marker_length_mm'] = self._marker_length_mm
        param_dict['dict_type'] = self._dict_type



    def dict_load(self, param_dict):
        self._squares = np.asarray(param_dict['squares'], dtype=int)
        self._square_length_pix = param_dict['square_length_pix']
        self._square_length_mm = param_dict['square_length_mm']
        self._marker_length_mm = param_dict['marker_length_mm']
        self._dict_type = param_dict['dict_type']



    def get_resolution_dpi(self):
        mm_per_inch = 25.4
        return (self._square_length_pix * mm_per_inch) / self._square_length_mm



    def _generate_board(self):
        aruco_dict = aruco.getPredefinedDictionary(self._dict_type)
        board = aruco.CharucoBoard(self._squares, self._square_length_mm,
            self._marker_length_mm, aruco_dict)
        return board



    def generate_image(self):
        board = self._generate_board()
        size_pixels = self._squares * self._square_length_pix
        image = board.generateImage(size_pixels)
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)



    def plot2d(self):
        image = self.generate_image()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        ax.set_title(f'squares {self._squares}, shape {image.shape}, dpi {self.get_resolution_dpi():.0f}')
        plt.show()



    def generate_mesh(self):
        image = self.generate_image()
        pixel_size = self._square_length_mm / self._square_length_pix
        return mesh_generate_image(image, pixel_size=pixel_size)



    def generate_screen(self):
        dimensions = self._squares * self._square_length_mm
        image = self.generate_image()
        return Screen(dimensions, image)



    def plot3d(self):
        cs_size = np.min(self._squares) * self._square_length_mm
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cs_size)
        mesh = self.generate_mesh()
        o3d.visualization.draw_geometries([cs, mesh])



    def detect_obj_img_points(self, images):
        board = self._generate_board()
        detector = aruco.CharucoDetector(board)
        all_obj_points = []
        all_img_points = []
        all_corners = []
        all_ids = []
        for i, image in enumerate(images):
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                detector.detectBoard(image)
            obj_points, img_points = board.matchImagePoints( \
                marker_corners, marker_ids)
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        return all_obj_points, all_img_points, all_corners, all_ids



    def calibrate(self, images, cam, flags=0, verbose=False):
        n = images.shape[0]
        image_shape = images.shape[1:3]
        # Extract object and image points
        obj_points, img_points, corners, ids = \
            self.detect_obj_img_points(images)
        # Calibrate camera
        reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
            cv2.calibrateCamera(obj_points, img_points, \
            image_shape, None, None, flags=flags)
        # Set intrincis
        cam.set_chip_size((images.shape[2], images.shape[1]))
        cam.set_camera_matrix(camera_matrix)
        cam.set_distortion(dist_coeffs)
        # Set extrinsics
        trafos = []
        for rvec, tvec in zip(rvecs, tvecs):
            trafo = Trafo3d(rodr=rvec, t=tvec) * self._T_CORR
            trafos.append(trafo)
        # If requested, visualize result
        if verbose:
            for i in range(n):
                annotated_image = cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR)
                aruco.drawDetectedCornersCharuco(annotated_image, corners[i],
                    ids[i], (255, 0, 255))
                rvec = trafos[i].get_rotation_rodrigues()
                tvec = trafos[i].get_translation()
                cv2.drawFrameAxes(annotated_image, camera_matrix, dist_coeffs, \
                    rvec, tvec, self._square_length_mm)
                annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

                fig = plt.figure()
                ax = plt.subplot(111)
                ax.imshow(annotated_image)
                plt.show()
        return trafos



    def estimate_pose(self, image, cam):
        pass
