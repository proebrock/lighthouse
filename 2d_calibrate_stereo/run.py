import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import copy
import glob
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.aruco_utils import CharucoBoard
from common.image_utils import image_load_multiple
from camsimlib.camera_model import CameraModel



def calculate_stereo_matrices(cam_left, cam_right):
    cam_right_to_cam_left = cam_right.get_pose().inverse() * cam_left.get_pose()
    t = cam_right_to_cam_left.get_translation()
    R = cam_right_to_cam_left.get_rotation_matrix()
    # Essential matrix E
    S = np.array([
        [ 0, -t[2], t[1] ],
        [ t[2], 0, -t[0] ],
        [ -t[1], t[0], 0 ],
    ])
    E = S @ R
    # Fundamental matrix F
    F = np.linalg.inv(cam_right.get_camera_matrix()).T @ E @ \
        np.linalg.inv(cam_left.get_camera_matrix())
    if not np.isclose(F[2, 2], 0.0):
        F = F / F[2, 2]
    return E, F


if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, '2d_calibrate_stereo')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load images
    pattern = os.path.join(data_dir, 'cam00_image??.png')
    images_left = image_load_multiple(pattern)
    pattern = os.path.join(data_dir, 'cam01_image??.png')
    images_right = image_load_multiple(pattern)

    # Load cameras
    filename = os.path.join(data_dir, 'cam00_image00.json')
    with open(filename) as f:
        params = json.load(f)
    cam_left = CameraModel()
    cam_left.dict_load(params['cam'])
    filename = os.path.join(data_dir, 'cam01_image00.json')
    with open(filename) as f:
        params = json.load(f)
    cam_right = CameraModel()
    cam_right.dict_load(params['cam'])
    cam_right_to_cam_left = cam_right.get_pose().inverse() * cam_left.get_pose()
    # Load board
    board = CharucoBoard()
    board.dict_load(params['board'])

    print('Running stereo calibration ...')
    flags = 0
#    flags |= cv2.CALIB_FIX_K1
#    flags |= cv2.CALIB_FIX_K2
    flags |= cv2.CALIB_FIX_K3
    flags |= cv2.CALIB_FIX_K4
    flags |= cv2.CALIB_FIX_K5
    flags |= cv2.CALIB_FIX_K6
    flags |= cv2.CALIB_ZERO_TANGENT_DIST
    #flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #flags |= cv2.CALIB_RATIONAL_MODEL
    cam_left_recalib, cam_right_recalib, cam_right_to_cam_left_estim, E, F, reprojection_error = \
        board.calibrate_stereo(images_left, images_right, flags)
    print(f'    Reprojection error is {reprojection_error:.2f}')

    print('Left camera')
    with np.printoptions(precision=1, suppress=True):
        print(f'    Real focal lengths {cam_left.get_focal_length()}')
        print(f'    Estm focal lengths {cam_left_recalib.get_focal_length()}')
        print(f'    Real principal points {cam_left.get_principal_point()}')
        print(f'    Estm principal points {cam_left_recalib.get_principal_point()}')
    with np.printoptions(precision=3, suppress=True):
        print(f'    Orig distortion {cam_left.get_distortion()}')
        print(f'    Estm distortion {cam_left_recalib.get_distortion()}')

    print('Right camera')
    with np.printoptions(precision=1, suppress=True):
        print(f'    Real focal lengths {cam_right.get_focal_length()}')
        print(f'    Estm focal lengths {cam_right_recalib.get_focal_length()}')
        print(f'    Real principal points {cam_right.get_principal_point()}')
        print(f'    Estm principal points {cam_right_recalib.get_principal_point()}')
    with np.printoptions(precision=3, suppress=True):
        print(f'    Real distortion {cam_right.get_distortion()}')
        print(f'    Estm distortion {cam_right_recalib.get_distortion()}')

    print('Camera pose cam_right_to_cam_left')
    print(f'    Real {str(cam_right_to_cam_left)}')
    print(f'    Estm {str(cam_right_to_cam_left_estim)}')
    dt, dr = cam_right_to_cam_left_estim.distance(cam_right_to_cam_left)
    print(f'    Errors: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')

    E2, F2 = calculate_stereo_matrices(cam_left, cam_right)
    E3, F3 = calculate_stereo_matrices(cam_left_recalib, cam_right_recalib)
    with np.printoptions(precision=3, suppress=True):
        print('\nEssential matrix')
        print(E)
        print(E2)
        print(E3)
        print('\nFundamental matrix')
        print(F)
        print(F2)
        print(F3)
