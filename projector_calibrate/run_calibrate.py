import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import cv2

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector



def calibrate(image_points, object_points, chip_size):
    num_boards = image_points.shape[0]
    assert image_points.shape[1] == object_points.shape[0]
    assert image_points.shape[2] == 2
    assert object_points.shape[1] == 3
    # Assemble obj/img points in a form accepted by calibration routine
    obj_points = []
    img_points = []
    for board_no in range(num_boards):
        mask = np.all(np.isfinite(image_points[board_no]), axis=1)
        obj_points.append(object_points[mask, :].astype(np.float32))
        img_points.append(image_points[board_no, mask, :].astype(np.float32))
    image_shape = chip_size[[1, 0]]
    #flags = 0
    flags = cv2.CALIB_ZERO_TANGENT_DIST | \
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.calibrateCamera(obj_points, img_points, \
        image_shape, None, None, flags=flags)
    cam_to_boards = []
    for rvec, tvec in zip(rvecs, tvecs):
        cam_to_boards.append(Trafo3d(rodr=rvec, t=tvec))
    return reprojection_error, camera_matrix, dist_coeffs, cam_to_boards



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'projector_calibrate')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load configuration
    filename = os.path.join(data_dir, 'projector.json')
    projector = ShaderProjector()
    projector.json_load(filename)
    cam_filenames = sorted(glob.glob(os.path.join(data_dir, 'cam??.json')))
    cams = []
    for i, filename in enumerate(cam_filenames):
        cam = CameraModel()
        cam.json_load(filename)
        cams.append(cam)
    filename = os.path.join(data_dir, 'points.npz')
    npz = np.load(filename)
    object_points = npz['object_points']
    cam_image_points = npz['cam_image_points']
    projector_image_points = npz['projector_image_points']


    # Calibrate cameras
    estimated_cams = []
    for cam_no in range(len(cams)):
        error, camera_matrix, dist_coeffs, cam_to_boards = \
            calibrate(cam_image_points[cam_no], object_points, cams[cam_no].get_chip_size())
        cam = CameraModel()
        cam.set_chip_size(cams[cam_no].get_chip_size())
        cam.set_camera_matrix(camera_matrix)
        cam.set_distortion(dist_coeffs)
        cam.set_pose(cam_to_boards[0].inverse())
        estimated_cams.append(cam)
    # Calibrate projector
    error, camera_matrix, dist_coeffs, proj_to_boards = \
        calibrate(projector_image_points, object_points, projector.get_chip_size())
    estimated_projector = ShaderProjector()
    estimated_projector.set_chip_size(projector.get_chip_size())
    estimated_projector.set_camera_matrix(camera_matrix)
    estimated_projector.set_distortion(dist_coeffs)
    estimated_projector.set_pose(proj_to_boards[0].inverse())
    # Make projector CS = world CS
    proj_to_board = estimated_projector.get_pose().inverse()
    for cam_no in range(len(cams)):
        # projector_to_board * board_to_cam
        estimated_cams[cam_no].set_pose(proj_to_board * estimated_cams[cam_no].get_pose())
    estimated_projector.set_pose(proj_to_board * estimated_projector.get_pose())
    # Show results of calibration
    for cam_no in range(len(cams)):
        print(cams[cam_no])
        print(estimated_cams[cam_no])
    print(projector)
    print(estimated_projector)

    # TODO: global optimization using initial estimates
