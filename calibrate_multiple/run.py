import cv2
import cv2.aruco as aruco
import glob
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d



def load_params(data_dir, cam_no, image_no):
    basename = os.path.join(data_dir, f'cam{cam_no:02d}_image{image_no:02d}')
    # Image parameters
    with open(basename + '.json', 'r') as f:
        params = json.load(f)
    board_squares = (params['board']['squares'][0], params['board']['squares'][1])
    board_square_length = params['board']['square_length']
    board_pose = Trafo3d(t=params['board']['pose']['t'], q=params['board']['pose']['q'])
    cam_pose = Trafo3d(t=params['cam']['camera_pose']['t'], q=params['cam']['camera_pose']['q'])
    cam_matrix = np.array([
        [ params['cam']['focal_length'][0], 0.0, params['cam']['principal_point'][0] ],
        [ 0.0, params['cam']['focal_length'][1], params['cam']['principal_point'][1] ],
        [ 0.0, 0.0, 1.0 ] ])
    return board_squares, board_square_length, board_pose, cam_pose, cam_matrix



def aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False):
    # Find corners in all images
    parameters = aruco.DetectorParameters_create()
    allCorners = []
    allIds = []
    imageSize = None
    images = [] # For debug view
    images_used = np.full(len(filenames), False)
    for i, fname in enumerate(filenames):
        print('Calibration using ' + fname + ' ...')
        # Load image
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageSize = gray.shape
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
        if charuco_corners is not None and charuco_corners.shape[0] >= 4:
            print(f'    Found {charuco_corners.shape[0]} corners.')
            allCorners.append(charuco_corners)
            allIds.append(charuco_ids)
            images.append(img)
            images_used[i] = True
        else:
            print('    Image rejected.')
    # Use corners to run global calibration
    flags = 0
    #flags |= cv2.CALIB_FIX_K1
    #flags |= cv2.CALIB_FIX_K2
    #flags |= cv2.CALIB_FIX_K3
    #flags |= cv2.CALIB_FIX_K4
    #flags |= cv2.CALIB_FIX_K5
    #flags |= cv2.CALIB_FIX_K6
    #flags |= cv2.CALIB_ZERO_TANGENT_DIST
    #flags |= cv2.CALIB_FIX_ASPECT_RATIO
    #flags |= cv2.CALIB_RATIONAL_MODEL
    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
        cv2.aruco.calibrateCameraCharuco(allCorners, allIds, \
        aruco_board, imageSize, None, None, flags=flags)
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



# Configuration
data_dir = 'a'
#data_dir = '/home/phil/pCloudSync/data/leafstring/calibrate_multiple'
if not os.path.exists(data_dir):
    raise Exception('Source directory does not exist.')
num_cams = 4
num_imgs = 10

# Create aruco board
board_squares, board_square_length, _, _, _ = \
    load_params(data_dir, 0, 0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
aruco_board = aruco.CharucoBoard_create(board_squares[0], board_squares[1],
                                  board_square_length,
                                  board_square_length/2.0, aruco_dict)

# Get nominal camera properties
nominal_cam_poses = []
nominal_cam_matrices = []
nominal_board_poses = []
for cam_no in range(num_cams):
    _, _, board_pose, cam_pose, cam_matrix = load_params(data_dir, cam_no, 0)
    nominal_cam_poses.append(cam_pose)
    nominal_cam_matrices.append(cam_matrix)
    board_poses = [ cam_pose.inverse() * board_pose ]
    for img_no in range(1, num_imgs):
        _, _, board_pose, _, _ = load_params(data_dir, cam_no, img_no)
        board_poses.append(cam_pose.inverse() * board_pose)
    nominal_board_poses.append(board_poses)

# Run calibrations
trafos = []
estimated_cam_matrices = []
for cam_no in range(num_cams):
    print(f' ------------- cam{cam_no} -------------')
    filenames = sorted(glob.glob(os.path.join(data_dir, f'cam{cam_no:02d}_image??_color.png')))
    images_used, reprojection_error, calib_trafos, camera_matrix, dist_coeffs = \
        aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
    print(f'Calibration done, reprojection error is {reprojection_error:.3f}')
    trafos.append(calib_trafos)
    estimated_cam_matrices.append(camera_matrix)


#print('###### Camera matrices ######')
#for cam_no in range(num_cams):
#    print(f' ------------- cam{cam_no} -------------')
#    print(nominal_cam_matrices[cam_no])
#    print(estimated_cam_matrices[cam_no])

#print('###### Single camera transformations ######')
#for cam_no in range(num_cams):
#    print(f' ------------- cam{cam_no} -------------')
#    for img_no in range(num_imgs):
##        print(nominal_board_poses[cam_no][img_no])
##        print(trafos[cam_no][img_no])
#        print(nominal_board_poses[cam_no][img_no].inverse() * trafos[cam_no][img_no])


# Identification of each single board pose


#for cam_no in range(num_cams):
#    for img_no in range(num_imgs):
#        trafos[cam_no][img_no] = trafos[0][img_no] * trafos[cam_no][img_no].inverse()
#
#for cam_no in range(num_cams):
#    print(f' ------------- cam{cam_no} -------------')
#    for img_no in range(num_imgs):
#        print(trafos[cam_no][img_no])
#    average, errors = Trafo3d.average_and_errors(trafos[cam_no])
#    print(average.inverse())
#    print(nominal_cam_poses[cam_no])

