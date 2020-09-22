import numpy as np
import cv2
import cv2.aruco as aruco

from MeshObject import MeshObject



class CharucoBoard(MeshObject):

    def __init__(self, squares, square_length):
        # Call base class constructor
        super(CharucoBoard, self).__init__()
        # Set up board parameters
        self.aruco_dict_index = aruco.DICT_4X4_50
        self.squares = np.asarray(squares)
        self.square_length = square_length
        self.marker_length = square_length / 2.0
        self.generate_board()



    def generate_board(self):
        # Generate ChArUco board
        aruco_dict = aruco.Dictionary_get(self.aruco_dict_index)
        board = aruco.CharucoBoard_create(self.squares[0], self.squares[1],
            self.square_length, self.marker_length, aruco_dict)
        # Draw image Marker in aruco_dict is 4x4, with margin 6x6;
        # marker length is half of square length, so total square has size of 12
        side_pixels = 12
        img_bw = board.draw((self.squares[0] * side_pixels, self.squares[1] * side_pixels))
        # According to documentation: "As in the GridBoard, the coordinate
        # system of the CharucoBoard is placed in the board plane with
        # the Z axis pointing out, and centered in the bottom left corner
        # of the board."
        # Meaning X axis to the right, Y axis up, Z pointing out
        if False:
            img = cv2.resize(img_bw, (0,0), fx=5.0, fy=5.0)
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # Convert grayscale image to color image
        img = np.zeros((img_bw.shape[0], img_bw.shape[1], 3))
        img[:,:,:] = img_bw[:,:,np.newaxis]
        self.generateFromImage(img, self.square_length/side_pixels)



    def dict_save(self, params):
        params['aruco_dict_index'] = self.aruco_dict_index
        params['squares'] = self.squares.tolist()
        params['square_length'] = self.square_length
        params['marker_length'] = self.marker_length
