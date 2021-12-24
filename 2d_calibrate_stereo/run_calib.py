import cv2
import copy
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
    cam_pose = cam.get_camera_pose()
    cam_matrix = cam.get_camera_matrix()
    cam_distortion = cam.get_distortion()
    return board_squares, board_square_length, board_pose, cam_pose, cam_matrix, cam_distortion


def aruco_find_corners(filenames, aruco_dict, aruco_board):
    # Find corners in all images
    parameters = aruco.DetectorParameters_create()
    all_corners = []
    all_ids = []
    image_size = None
    images = [] # For debug view
    images_used = np.full(len(filenames), False)
    for i, fname in enumerate(filenames):
        print('Calibration using ' + fname + ' ...')
        # Load image
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape
        else:
            assert image_size == gray.shape
        # Detect corners
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
            parameters=parameters)
        #aruco.drawDetectedMarkers(img, corners, ids)
        # Refine and interpolate corners
        corners, ids, rejected, recovered_ids = aruco.refineDetectedMarkers( \
            gray, aruco_board, corners, ids, rejected)
        charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco( \
            corners, ids, gray, aruco_board)
        aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        # Check if enough corners found
        if charuco_corners is not None and charuco_corners.shape[0] >= 6:
            print(f'    Found {charuco_corners.shape[0]} corners.')
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
            images.append(img)
            images_used[i] = True
        else:
            print('    Image rejected.')
    return images, images_used, all_corners, all_ids, image_size



def aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False):
    # Find corners in all images
    images, images_used, all_corners, all_ids, image_size = aruco_find_corners(filenames, aruco_dict, aruco_board)
    # Use corners to run global calibration
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
    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, \
        aruco_board, image_size, None, None, flags=flags)
    calib_trafos = []
    for r, t in zip(rvecs, tvecs):
        calib_trafos.append(Trafo3d(t=t, rodr=r))
    if verbose:
        # Visualize boards with features that have been found
        for i in range(len(images)):
            img = images[i]
            aruco.drawAxis(img, camera_matrix, dist_coeffs, \
                rvecs[i], tvecs[i], aruco_board.getSquareLength())
            img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
            cv2.imshow(f'image{i:02d}', img)
            key = cv2.waitKey(0) & 0xff
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
    return images_used, reprojection_error, calib_trafos, camera_matrix, dist_coeffs



def aruco_generate_object_points(board_square_length, board_squares):
    x = np.arange(1, board_squares[0])
    y = np.arange(1, board_squares[1])
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.ravel(), Y.ravel())).T
    points = board_square_length * points
    points = points[:, np.newaxis, :]
    points = points.tolist()
    return points



def aruco_calibrate_stereo(filenames_l, filenames_r, board_square_length, board_squares, \
        aruco_dict, aruco_board, verbose=False):
    assert len(filenames_l) == len(filenames_r)
    images_l, images_used_l, all_corners_l, all_ids_l, image_size_l = \
        aruco_find_corners(filenames_l, aruco_dict, aruco_board)
    images_r, images_used_r, all_corners_r, all_ids_r, image_size_r = \
        aruco_find_corners(filenames_r, aruco_dict, aruco_board)
    assert image_size_l == image_size_r
    image_size = image_size_l

    obj_points_board = aruco_generate_object_points(board_square_length, board_squares)
    # all_ids shape: num images (12) x num corners x 1 x 2
    # all_corners shape: num images (12) x num corners x 1
    obj_points = None
    img_points_l = None
    img_points_r = None
    flags = 0
#    reprojection_error, camera_matrix_l, dist_coeffs_l, camera_matrix_r, dist_coeffs_r, R, T, E, F = \
#        cv2.stereoCalibrate(obj_points, img_points_l, img_points_r, None, None, None, None, image_size, flags=flags)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    # Configuration
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/2d_calibrate_multiple'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')
    num_cams = 2
    num_imgs = 12

    # Create aruco board
    board_squares, board_square_length, _, _, _, _ = \
        load_params(data_dir, 0, 0)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    aruco_board = aruco.CharucoBoard_create(board_squares[0], board_squares[1],
                                      board_square_length,
                                      board_square_length/2.0, aruco_dict)

    # Get nominal camera properties
    nominal_board_poses = []
    nominal_cam_poses = []
    nominal_cam_matrices = []
    nominal_cam_distortions = []
    for cam_no in range(num_cams):
        _, _, board_pose, cam_pose, cam_matrix, cam_distortion = load_params(data_dir, cam_no, 0)
        nominal_cam_poses.append(cam_pose)
        nominal_cam_matrices.append(cam_matrix)
        board_poses = [ cam_pose.inverse() * board_pose ]
        for img_no in range(1, num_imgs):
            _, _, board_pose, _, _, _ = load_params(data_dir, cam_no, img_no)
            board_poses.append(cam_pose.inverse() * board_pose)
        nominal_board_poses.append(board_poses)
        nominal_cam_distortions.append(cam_distortion)

    # Run calibrations
    print('Running calibrations for each camera separately ...')
    trafos = []
    estimated_cam_matrices = []
    estimated_cam_distortions = []
    for cam_no in range(num_cams):
        print(f' ------------- cam{cam_no} -------------')
        filenames = sorted(glob.glob(os.path.join(data_dir, f'cam{cam_no:02d}_image??_color.png')))
        images_used, reprojection_error, calib_trafos, camera_matrix, dist_coeffs = \
            aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
        if not np.all(images_used):
            raise Exception('There were errors using all images for calibration; handling of this is not implemented')
        print(f'Calibration done, reprojection error is {reprojection_error:.3f}')
        trafos.append(calib_trafos)
        estimated_cam_matrices.append(camera_matrix)
        estimated_cam_distortions.append(dist_coeffs)

    # Use coordinate system of camera 0 as reference coordinate system!
    # trafo[i][j] contains the transformation FROM cam_i TO board in image j:
    # cam{i}_T_board{j}
    #
    # So to calculate the transformation from cam0 to cam1 just using
    # the first board pose and the images from those two cameras:
    print(trafos[0][0] * trafos[1][0].inverse())



    print('\n###### Camera matrices ######')
    for cam_no in range(num_cams):
        print(f' ------------- cam{cam_no} -------------')
        with np.printoptions(precision=1, suppress=True):
            print(nominal_cam_matrices[cam_no])
            print(estimated_cam_matrices[cam_no])

    print('\n###### Camera distortions ######')
    for cam_no in range(num_cams):
        print(f' ------------- cam{cam_no} -------------')
        with np.printoptions(precision=3, suppress=True):
            print(nominal_cam_distortions[cam_no])
            print(estimated_cam_distortions[cam_no])

    print('\n###### Single camera transformations ######')
    for cam_no in range(num_cams):
        print(f' ------------- cam{cam_no} -------------')
        for img_no in range(num_imgs):
    #        print(nominal_board_poses[cam_no][img_no], '<')
    #        print(trafos[cam_no][img_no])
            print(nominal_board_poses[cam_no][img_no].inverse() * trafos[cam_no][img_no])

    # We run over all images and calculate the transformations relative to cam0
    cam0_trafo = copy.deepcopy(trafos[0])
    for cam_no in range(num_cams):
        for img_no in range(num_imgs):
            trafos[cam_no][img_no] = cam0_trafo[img_no] * trafos[cam_no][img_no].inverse()

    # Make all nominal camera poses relative to cam0
    cam0_trafo = copy.deepcopy(nominal_cam_poses[0])
    for cam_no in range(num_cams):
        nominal_cam_poses[cam_no] = cam0_trafo.inverse() * nominal_cam_poses[cam_no]

    # Average over all num_imgs transformations for each camera
    # to get best estimate of camera poses relative to camera 0
    estimated_cam_poses = []
    for cam_no in range(num_cams):
        average, errors = Trafo3d.average_and_errors(trafos[cam_no])
        estimated_cam_poses.append(average)
    #    print(nominal_cam_poses[cam_no])
    #    print(average)
    #    print(f'error: dt={np.mean(errors[:,0]):.1f}, dr={np.mean(np.rad2deg(errors[:,1])):.2f} deg')

    print('\n###### Camera poses ######')
    for cam_no in range(num_cams):
        print(f' ------------- cam{cam_no} -------------')
        print(nominal_cam_poses[cam_no])
        print(estimated_cam_poses[cam_no])
        dt, dr = nominal_cam_poses[cam_no].distance(estimated_cam_poses[cam_no])
        print(f'Error: dt={dt:.1f}, dr={np.rad2deg(dr):.2f} deg')


    # TODO:
    # * add/compare to 2d_calibrate_multiple
    # * remove detailed output
    # * add new function aruco_calibrate_stereo -> pose left-right, E, F
    #   (https://stackoverflow.com/questions/64612924/opencv-stereocalibration-of-two-cameras-using-charuco)
    # * manually calculate E, F from pose and compare with result (new function)

    cam_no_l = 0
    cam_no_r = 1
    filenames_l = sorted(glob.glob(os.path.join(data_dir, f'cam{cam_no_l:02d}_image??_color.png')))
    filenames_r = sorted(glob.glob(os.path.join(data_dir, f'cam{cam_no_r:02d}_image??_color.png')))
    print('Running stereo calibration ...')
    aruco_calibrate_stereo(filenames_l, filenames_r, board_square_length, board_squares, \
        aruco_dict, aruco_board, verbose=False)
