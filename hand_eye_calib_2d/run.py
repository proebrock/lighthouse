import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import os
import sys
import open3d as o3d
import time

from scipy.optimize import minimize

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from common.aruco_utils import charuco_calibrate



def toParam(base_to_board, flange_to_cam):
    x = []
    x.extend(base_to_board.get_translation())
    x.extend(base_to_board.get_rotation_rodrigues())
    x.extend(flange_to_cam.get_translation())
    x.extend(flange_to_cam.get_rotation_rodrigues())
    return np.array(x)

def fromParam(x):
    base_to_board = Trafo3d(t=x[0:3], rodr=x[3:6])
    flange_to_cam = Trafo3d(t=x[6:9], rodr=x[9:12])
    return base_to_board, flange_to_cam

def obj_fun(x, base_to_flanges, cams_to_board):
    base_to_board, flange_to_cam = fromParam(x)
    n = len(base_to_flanges)
    dtsum = 0
    drsum = 0
    for i in range(n):
        base_to_board_calculated = base_to_flanges[i] * \
            flange_to_cam * cams_to_board[i]
        dt, dr = base_to_board_calculated.distance(base_to_board)
        dtsum += dt
        drsum += dr
    # Weight here the translational and rotational errors
    weight = 1.0
    return dtsum + weight * np.rad2deg(drsum)

def hand_eye_calibrate_optim(base_to_flanges, cams_to_board):
    # Initial values
    base_to_board_0 = Trafo3d()
    flange_to_cam_0 = Trafo3d()
    x0 = toParam(base_to_board_0, flange_to_cam_0)

    print('\nRunning optimization, please stand by ...')
    options={ 'maxiter': 200000, 'maxfev': 200000, 'adaptive': True }
    tic = time.monotonic()
    result = minimize(obj_fun, x0, args=(base_to_flanges, cams_to_board),
                      method='Nelder-Mead', options=options)
    toc = time.monotonic()
    print(f'Done. Optimization took {toc-tic:.1f}s.')
    if result.success:
        base_to_board, flange_to_cam = fromParam(result.x)
        return base_to_board, flange_to_cam
    else:
        return None, None



def hand_eye_calibrate_opencv(base_to_flanges, cams_to_board):
    # "flange" is called "gripper" in this notation
    R_gripper2base = []
    t_gripper2base = []
    for T in base_to_flanges:
        R = T.get_rotation_matrix()
        t = T.get_translation()
        R_gripper2base.append(R)
        t_gripper2base.append(t)
    # "board" is called "target" in this notation
    R_target2cam = []
    t_target2cam = []
    for T in cams_to_board:
        R = T.get_rotation_matrix()
        t = T.get_translation()
        R_target2cam.append(R)
        t_target2cam.append(t)
    # Calculate calibration using certain method
#    method = cv2.CALIB_HAND_EYE_TSAI
    method = cv2.CALIB_HAND_EYE_PARK
#    method = cv2.CALIB_HAND_EYE_HORAUD
#    method = cv2.CALIB_HAND_EYE_ANDREFF
#    method = cv2.CALIB_HAND_EYE_DANIILIDIS
    print('\nRunning OpenCV hand-eye-calibration ...')
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye( \
        R_gripper2base, t_gripper2base, \
        R_target2cam, t_target2cam, \
        method=method)
    print('Done.')
    flange_to_cam = Trafo3d(t=t_cam2gripper, mat=R_cam2gripper)
    # Calculate the by-product of the calibration from the result above
    base_to_board = base_to_flanges[0] * flange_to_cam * cams_to_board[0]
    return base_to_board, flange_to_cam



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

    #
    # Read parameters that were used to generate calibration images;
    # this is the ground truth to compare the calibration results with
    #
    aruco_dict = None
    aruco_board = None
    parameters = aruco.DetectorParameters_create()
    cam_matrix = None
    cam_dist = None
    cam_trafos = []

    base_to_flanges = []
    base_to_board_real = None # Real value, should match calibration
    flange_to_cam_real = None # Real value, should match calibration

    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    for fname in filenames:
        with open(fname) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cam_trafos.append(cam.get_pose())
        base_to_flange = Trafo3d(t=params['base_to_flange']['t'],
                                 q=params['base_to_flange']['q'])
        base_to_flanges.append(base_to_flange)
        if aruco_dict is None:
            # Assumption is that all images show the same aruco board
            # taken with the same camera
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            aruco_board = aruco.CharucoBoard_create( \
                params['board']['squares'][0], params['board']['squares'][1],
                params['board']['square_length'], params['board']['square_length'] / 2.0,
                aruco_dict)
            cam_matrix = cam.get_camera_matrix()
            cam_dist = cam.get_distortion()
            # Those trafos are the same for all images
            base_to_board_real = Trafo3d(t=params['base_to_board']['t'],
                                    q=params['base_to_board']['q'])
            flange_to_cam_real = Trafo3d(t=params['flange_to_cam']['t'],
                                    q=params['flange_to_cam']['q'])



    #
    # Run camera calibration
    #
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images_used, reprojection_error, cams_to_board, camera_matrix, dist_coeffs = \
        charuco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
    for index in np.where(~images_used)[0]:
        del base_to_flanges[index]
    assert(len(cams_to_board) == len(base_to_flanges))



    #
    # Calculating the hand-eye calibration
    #
    # Transformations involved:
    #                                         (should be equal)
    # base_to_flanges * flange_to_cam * cams_to_board = base_to_board
    # (multiple, from   (single,        (multiple,      (single,
    # robot prg)        unknown)        from calib)     unknown by product)
    #
    base_to_board_estim, flange_to_cam_estim = \
        hand_eye_calibrate_optim(base_to_flanges, cams_to_board)
    print('------------ base_to_board ------------')
    print(base_to_board_real)
    print(base_to_board_estim)
    dt, dr = base_to_board_estim.distance(base_to_board_real)
    print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
    print('------------ flange_to_cam ------------')
    print(flange_to_cam_real)
    print(flange_to_cam_estim)
    dt, dr = flange_to_cam_estim.distance(flange_to_cam_real)
    with np.printoptions(precision=2, suppress=True):
        print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')

    base_to_board_estim, flange_to_cam_estim = \
        hand_eye_calibrate_opencv(base_to_flanges, cams_to_board)
    print('------------ base_to_board ------------')
    print(base_to_board_real)
    print(base_to_board_estim)
    dt, dr = base_to_board_estim.distance(base_to_board_real)
    print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
    print('------------ flange_to_cam ------------')
    print(flange_to_cam_real)
    print(flange_to_cam_estim)
    dt, dr = flange_to_cam_estim.distance(flange_to_cam_real)
    with np.printoptions(precision=2, suppress=True):
        print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
