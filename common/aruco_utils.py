import cv2
import cv2.aruco as aruco
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt



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



    def __str__(self):
        pass



    def dict_save(self, param_dict):
        pass



    def dict_load(self, param_dict):
        pass



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
        return board.generateImage(size_pixels)



    def plot2d(self):
        image = self.generate_image()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image, cmap='gray')
        ax.set_title(f'squares {self._squares}, shape {image.shape}, dpi {self.get_resolution_dpi():.0f}')
        plt.show()



    def generate_mesh(self):
        pass



    def plot3d(self):
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self._square_length_mm)
        mesh = self.generate_mesh()
        o3d.visualization.draw_geometries([cs, mesh])



    def detect_obj_img_points(self, images):
        board = self._generate_board()
        detector = aruco.CharucoDetector(board)
        all_obj_points = []
        all_img_points = []
        for i, image in enumerate(images):
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                detector.detectBoard(image)
            print(f'Found {marker_corners.shape[0]} corners in image {i}')
            obj_points, img_points = board.matchImagePoints(marker_corners, marker_ids)
            all_obj_points.append(obj_points)
            all_img_points.append(img_points)
        return all_obj_points, all_img_points



    def annotate_image(self, image, corners, ids, cam):
        aruco.drawDetectedCornersCharuco(image, corners, ids, (255, 0, 255))
        camera_matrix = cam.get_camera_matrix()
        dist_coeffs = cam.get_distortion()
        rvec, tvec = cam.get_rvec_tvec()
        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, \
            rvec, tvec, self._square_length_mm)




board = CharucoBoard((5, 7), 80, 20, 10)
board.plot2d()