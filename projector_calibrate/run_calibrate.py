import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import cv2

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.projective_geometry import ProjectiveGeometry
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



def trafo_to_x(pose : Trafo3d):
    return np.hstack([ pose.get_translation(), 100.0 * pose.get_rotation_rodrigues()])



def x_to_trafo(pose : Trafo3d, x):
    pose.set_translation(x[0:3])
    pose.set_rotation_rodrigues(x[3:6] / 100.0)
    return 6



def projective_geometry_intrinsics_to_x(pg : ProjectiveGeometry):
    focal_length = pg.get_focal_length()
    get_principal_point = pg.get_principal_point()
    if True:
        return np.hstack([ focal_length, get_principal_point ])
    else:
        distortion = pg.get_distortion()
        return np.stack([ focal_length, get_principal_point, 10000.0 * distortion[0:5] ])



def x_to_projective_geometry_intrinsics(pg : ProjectiveGeometry, x):
    pg.set_focal_length(x[0:2])
    pg.set_principal_point(x[2:4])
    if True:
        return 4
    else:
        pg.set_distortion(x[4:9] / 10000.0)
        return 9



def projective_geometry_extrinsics_to_x(pg : ProjectiveGeometry):
    return trafo_to_x(pg.get_pose())



def x_to_projective_geometry_extrinsics(pg : ProjectiveGeometry, x):
    pose = pg.get_pose()
    i = x_to_trafo(pose, x)
    pg.set_pose(pose)
    return i



def param_to_x(pg_list : list[ProjectiveGeometry], board_poses : list[Trafo3d]):
    x = []
    x.append(projective_geometry_intrinsics_to_x(pg_list[0]))
    for i in range(1, len(pg_list)):
        x.append(projective_geometry_extrinsics_to_x(pg_list[i]))
        x.append(projective_geometry_intrinsics_to_x(pg_list[i]))
    for i in range(len(board_poses)):
        x.append(trafo_to_x(board_poses[i]))
    return np.hstack(x)



def x_to_param(pg_list: list[ProjectiveGeometry], board_poses : list[Trafo3d], x):
    index = 0
    index += x_to_projective_geometry_intrinsics(pg_list[0], x[index:])
    for i in range(1, len(pg_list)):
        index += x_to_projective_geometry_extrinsics(pg_list[i], x[index:])
        index += x_to_projective_geometry_intrinsics(pg_list[i], x[index:])
    for i in range(len(board_poses)):
        index += x_to_trafo(board_poses[i], x[index:])
    return index



def objfun(x, pg_list, board_poses, object_points, image_points):
    x_to_param(pg_list, board_poses, x)
    residuals = []
    for pg_no in range(image_points.shape[0]):
        for board_no in range(image_points.shape[1]):
            P = board_poses[board_no] * object_points
            p = pg_list[pg_no].scene_to_chip(P)
            p = p[:, 0:2] # Omit distance
            dist = p - image_points[pg_no, board_no, :, :]
            mask = np.all(np.isfinite(dist), axis=1)
            dist = dist[mask, :]
            dist = np.linalg.norm(dist, axis=1)
            residuals.append(dist)
    return np.hstack(residuals)



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
    print('### Preliminary calibration results\n')
    print(projector)
    print(estimated_projector)
    for cam_no in range(len(cams)):
        print(cams[cam_no])
        print(estimated_cams[cam_no])


    # Join projector and cameras in a list of projective geometries
    pg_list = [ estimated_projector, *estimated_cams ]
    # Join projector and camera image points in common image point structure
    pg_image_points = np.zeros((1 + len(cams), *projector_image_points.shape))
    pg_image_points[0, :, :, :] = projector_image_points
    for cam_no in range(len(cams)):
        pg_image_points[cam_no + 1, :, :, :] = cam_image_points[cam_no, :, :, :]
    # Create initial values for optimization
    x0 = param_to_x(pg_list, proj_to_boards)
    # Get residuals of initial estimate
    residuals0 = objfun(x0, pg_list, proj_to_boards, object_points, pg_image_points)
    # Run numerical optimization
    tic = time.monotonic()
    result = least_squares(objfun, x0, args=(pg_list, proj_to_boards, object_points, pg_image_points))
    toc = time.monotonic()
    print(f'Optimizaton image took {(toc - tic):.1f}s')
    if not result.success:
        raise Exception(f'Numerical optimization failed: {result}')
    residuals = objfun(result.x, pg_list, proj_to_boards, object_points, pg_image_points)


    # Plot residuals
    _, ax = plt.subplots()
    ax.plot(residuals0, label='initial')
    ax.plot(residuals, label='final')
    ax.grid()
    ax.legend()
    plt.show()


    # Show results of calibration
    x_to_param(pg_list, proj_to_boards, result.x)
    print('\n### Final calibration results\n')
    print(projector)
    print(pg_list[0])
    for cam_no in range(len(cams)):
        print(cams[cam_no])
        print(pg_list[cam_no + 1])
