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



def calibrate(image_points, object_points, chip_size, estimate_distortion):
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
    if estimate_distortion:
        flags = 0
    else:
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



def projective_geometry_intrinsics_to_x(pg : ProjectiveGeometry, estimate_distortion):
    focal_length = pg.get_focal_length()
    get_principal_point = pg.get_principal_point()
    if estimate_distortion:
        distortion = pg.get_distortion()
        return np.hstack([ focal_length, get_principal_point, 10000.0 * distortion[0:5] ])
    else:
        return np.hstack([ focal_length, get_principal_point ])



def x_to_projective_geometry_intrinsics(pg : ProjectiveGeometry, estimate_distortion, x):
    pg.set_focal_length(x[0:2])
    pg.set_principal_point(x[2:4])
    if estimate_distortion:
        pg.set_distortion(x[4:9] / 10000.0)
        return 9
    else:
        return 4



def projective_geometry_extrinsics_to_x(pg : ProjectiveGeometry):
    return trafo_to_x(pg.get_pose())



def x_to_projective_geometry_extrinsics(pg : ProjectiveGeometry, x):
    pose = pg.get_pose()
    i = x_to_trafo(pose, x)
    pg.set_pose(pose)
    return i



def param_to_x(pg_list : list[ProjectiveGeometry], estimate_distortions,
    board_poses : list[Trafo3d]):
    x = []
    x.append(projective_geometry_intrinsics_to_x(pg_list[0],
        estimate_distortions[0]))
    for i in range(1, len(pg_list)):
        x.append(projective_geometry_extrinsics_to_x(pg_list[i]))
        x.append(projective_geometry_intrinsics_to_x(pg_list[i],
            estimate_distortions[i]))
    for i in range(len(board_poses)):
        x.append(trafo_to_x(board_poses[i]))
    return np.hstack(x)



def x_to_param(pg_list: list[ProjectiveGeometry], estimate_distortions,
    board_poses : list[Trafo3d], x):
    index = 0
    index += x_to_projective_geometry_intrinsics(pg_list[0],
        estimate_distortions[0], x[index:])
    for i in range(1, len(pg_list)):
        index += x_to_projective_geometry_extrinsics(pg_list[i], x[index:])
        index += x_to_projective_geometry_intrinsics(pg_list[i],
            estimate_distortions[i], x[index:])
    for i in range(len(board_poses)):
        index += x_to_trafo(board_poses[i], x[index:])
    return index



def objfun(x, pg_list, estimate_distortions, board_poses, object_points, image_points):
    x_to_param(pg_list, estimate_distortions, board_poses, x)
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


    projective_geometries = [ projector ]
    projective_geometries.extend(cams)
    # Create new projective geometries
    estimated_projective_geometries = [ ShaderProjector() ]
    for cam_no in range(len(cams)):
        estimated_projective_geometries.append(CameraModel())
    # Join projector and camera image points in common image point structure
    image_points = np.zeros((1 + len(cams), *projector_image_points.shape))
    image_points[0, :, :, :] = projector_image_points
    for cam_no in range(len(cams)):
        image_points[cam_no + 1, :, :, :] = cam_image_points[cam_no, :, :, :]


    # Calibrate cameras
    estimate_distortions = [ True, False, False ]
    board_poses = []
    for pg_no in range(len(estimated_projective_geometries)):
        # Transfer chip size from original
        chip_size = projective_geometries[pg_no].get_chip_size()
        error, camera_matrix, dist_coeffs, pg_to_boards = \
            calibrate(image_points[pg_no], object_points,
                chip_size, estimate_distortions[pg_no])
        pg = estimated_projective_geometries[pg_no]
        pg.set_chip_size(chip_size)
        pg.set_camera_matrix(camera_matrix)
        pg.set_distortion(dist_coeffs)
        pg.set_pose(pg_to_boards[0].inverse())
        if pg_no == 0:
            board_poses = pg_to_boards
    # Make first projective geometry CS = world CS
    pg0_to_board = estimated_projective_geometries[0].get_pose().inverse()
    for pg_no in range(len(estimated_projective_geometries)):
        pose = estimated_projective_geometries[pg_no].get_pose()
        estimated_projective_geometries[pg_no].set_pose(pg0_to_board * pose)

    # Show results of calibration
    print('### Preliminary calibration results\n')
    for pg, epg in zip(projective_geometries, estimated_projective_geometries):
        print(pg)
        print(epg)


    # Create initial values for optimization
    x0 = param_to_x(estimated_projective_geometries,
        estimate_distortions, board_poses)
    # Get residuals of initial estimate
    residuals0 = objfun(x0, estimated_projective_geometries,
        estimate_distortions, board_poses, object_points, image_points)
    residuals0_rms = np.sqrt(np.mean(np.square(residuals0)))
    # Run numerical optimization
    print('\nRunning global optimization ...')
    tic = time.monotonic()
    result = least_squares(objfun, x0, args=(estimated_projective_geometries,
        estimate_distortions, board_poses, object_points, image_points))
    toc = time.monotonic()
    print(f'Optimizaton image took {(toc - tic):.1f}s')
    if not result.success:
        raise Exception(f'Numerical optimization failed: {result}')
    residuals = objfun(result.x, estimated_projective_geometries,
        estimate_distortions, board_poses, object_points, image_points)
    residuals_rms = np.sqrt(np.mean(np.square(residuals)))


    # Plot residuals
    _, ax = plt.subplots()
    ax.plot(residuals0, label=f'initial, RMS={residuals0_rms:.3f} pix')
    ax.plot(residuals,  label=f'final, RMS={residuals_rms:.3f} pix')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Residual index')
    ax.set_ylabel('Residual (pixel)')
    plt.show()


    # Show results of calibration
    x_to_param(estimated_projective_geometries, estimate_distortions,
        board_poses, result.x)
    print('\n### Final calibration results\n')
    for pg, epg in zip(projective_geometries, estimated_projective_geometries):
        print(pg)
        print(epg)
