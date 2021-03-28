import cv2
import cv2.aruco as aruco
import glob
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def load_params(data_dir, cam_no, image_no):
    basename = os.path.join(data_dir, f'cam{cam_no:02d}_image{image_no:02d}')
    # Image parameters
    with open(basename + '.json', 'r') as f:
        params = json.load(f)
    board_squares = (params['board']['squares'][0], params['board']['squares'][1])
    board_square_length = params['board']['square_length']
    board_pose = Trafo3d(t=params['board']['pose']['t'], q=params['board']['pose']['q'])
    cam = CameraModel()
    cam.dict_load(params['cam'])
    return board_squares, board_square_length, board_pose, cam



def aruco_get_corners(filename, aruco_dict, aruco_board, verbose=False):
    # Load image
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect corners
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
        parameters=parameters)
    #aruco.drawDetectedMarkers(img, corners, ids)
    # Refine and interpolate corners
    corners, ids, rejected, recovered_ids = aruco.refineDetectedMarkers( \
        gray, aruco_board, corners, ids, rejected)
    charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco( \
        corners, ids, gray, aruco_board)
    aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

    if verbose:
        img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return charuco_corners[:,0,:], charuco_ids[:,0]



def generate_board_scene_points(board_square_length, board_squares):
    """ Generate 3D scene coordinates of the corners of aruco board
    IDs first increase in X direction, the in Y direction, Z coordinates are all zero
    :param board_square_length: Square length
    :param board_squares: Number of squares in X/Y direction
    :return: Scene points, IDs
    """
    x = board_square_length * np.arange(1, board_squares[0])
    y = board_square_length * np.arange(1, board_squares[1])
    x, y = np.meshgrid(x, y)
    P = np.vstack((x.ravel(), y.ravel(), np.zeros(x.size))).T
    ids = np.arange(x.size)
    return P, ids



def generate_board_chip_points(board_square_length, board_squares, cam, board_poses):
    """ Generate 2D chip points of the corners of the aruco board
    :param board_square_length: Square length
    :param board_squares: Number of squares in X/Y direction
    :param cam: Camera to take the image with
    :param board_poses: Pose of the board
    :return: Chip points, IDs
    """
    P, ids = generate_board_scene_points(board_square_length, board_squares)
    p = cam.scene_to_chip(board_poses * P)[:,0:2]
    return p, ids



def match_points(points0, ids0, points1, ids1):
    pass



def calculate_homography(points0, points1):
    # cv2.findHomography
    pass



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    # Configuration
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/leafstring/2d_calibrate_multiple'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')
    num_cams = 4
    num_imgs = 12

    # Create aruco board
    board_squares, board_square_length, _, _ = \
        load_params(data_dir, 0, 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_board = aruco.CharucoBoard_create(board_squares[0], board_squares[1],
                                      board_square_length,
                                      board_square_length/2.0, aruco_dict)

    # Load cameras, all with proper poses, pose of cam[0] is world CS
    cams = []
    for cam_no in range(num_cams):
        _, _, _, cam = load_params(data_dir, cam_no, 0)
        cams.append(cam)

    # Load board poses; pose of cam[0] is world CS, so get board poses relative to cam[0]
    board_poses = []
    for img_no in range(num_imgs):
        _, _, board_pose, _ = load_params(data_dir, 0, img_no)
        board_poses.append(cams[0].get_camera_pose().inverse() * board_pose)



    p, ids = generate_board_chip_points(board_square_length, board_squares,
                                   cams[1], board_poses[0])
    print(p)

    filename = os.path.join(data_dir, f'cam00_image00_color.png')
    points0, ids0 = aruco_get_corners(filename, aruco_dict, aruco_board, verbose=False)

    filename = os.path.join(data_dir, f'cam01_image00_color.png')
    points1, ids1 = aruco_get_corners(filename, aruco_dict, aruco_board, verbose=False)
    print(points1)
