import matplotlib.pyplot as plt
import json
import numpy as np
import os
import sys
import cv2
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.aruco_utils import CharucoBoard
from common.image_utils import image_load_multiple
from camsimlib.camera_model import CameraModel



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'hand_eye_calib_2d')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load images
    pattern = os.path.join(data_dir, 'image??.png')
    images = image_load_multiple(pattern)

    # Load board
    filename = os.path.join(data_dir, 'image00.json')
    with open(filename) as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    cam.set_pose(Trafo3d())
    board = CharucoBoard()
    board.dict_load(params['board'])

    print('Running intrinsics calibration ...')
    #flags = 0
    flags = cv2.CALIB_ZERO_TANGENT_DIST | \
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    cam_recalib, cam_to_boards_estim, reprojection_error = \
        board.calibrate_intrinsics(images, flags=flags)
    print(f'    Reprojection error is {reprojection_error:.2f}')

    print('Comparing results ...')
    with np.printoptions(precision=1, suppress=True):
        print(f'    Real focal lengths {cam.get_focal_length()}')
        print(f'    Estm focal lengths {cam_recalib.get_focal_length()}')
        print(f'    Real principal points {cam.get_principal_point()}')
        print(f'    Estm principal points {cam_recalib.get_principal_point()}')
    with np.printoptions(precision=3, suppress=True):
        print(f'    Real distortion {cam.get_distortion()}')
        print(f'    Estm distortion {cam_recalib.get_distortion()}')

