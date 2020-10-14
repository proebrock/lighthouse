import cv2
import cv2.aruco as aruco
import glob
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.charuco_board import CharucoBoard
from camsimlib.scene_visualizer import SceneVisualizer



#
# Read parameters that were used to generate calibration images;
# this is the ground truth to compare the calibration results with
#

data_dir = 'a'
if not os.path.exists(data_dir):
    raise Exception('Source directory does not exist.')

aruco_dict = None
board = None
parameters = aruco.DetectorParameters_create()
cam_matrix = None
cam_dist = None
cam_trafos = []

for fname in sorted(glob.glob(os.path.join(data_dir, '*.json'))):
    with open(fname) as f:
        params = json.load(f)
    T = Trafo3d(t=params['cam']['camera_pose']['t'], q=params['cam']['camera_pose']['q'])
    cam_trafos.append(T)
    if aruco_dict is None:
        # Assumption is that all images show the same aruco board
        aruco_dict = aruco.Dictionary_get(params['board']['aruco_dict_index'])
        board = aruco.CharucoBoard_create( \
            params['board']['squares'][0], params['board']['squares'][1],
            params['board']['square_length'], params['board']['marker_length'],
            aruco_dict)
        cam_matrix = np.array([
            [ params['cam']['focal_length'][0], 0.0, params['cam']['principal_point'][0] ],
            [ 0.0, params['cam']['focal_length'][1], params['cam']['principal_point'][1] ],
            [ 0.0, 0.0, 1.0 ] ])
        cam_dist = params['cam']['distortion']


#
# Run the OpenCV calibration on the calibration images
#

allCorners = []
allIds = []

imageSize = None
images = []

for fname in sorted(glob.glob(os.path.join(data_dir, '*.png'))):
    print('Calibration using ' + fname + ' ...')

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageSize = gray.shape

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
        parameters=parameters)

    #aruco.drawDetectedMarkers(img, corners, ids)

    corners, ids, rejected, recovered_ids = aruco.refineDetectedMarkers( \
        gray, board, corners, ids, rejected)

    charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco( \
        corners, ids, gray, board)

    aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

    if charuco_corners is not None and charuco_corners.shape[0] >= 4:
        print(f'    Found {charuco_corners.shape[0]} corners.')
        allCorners.append(charuco_corners)
        allIds.append(charuco_ids)
        images.append(img)
    else:
        print('    Image rejected.')

print('Calculating calibration ...')
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
    board, imageSize, None, None, flags=flags)
calib_trafos = []
for r, t in zip(rvecs, tvecs):
    calib_trafos.append(Trafo3d(t=t, rodr=r))
print(f'Calibration done, reprojection error is {reprojection_error:.2f}')
print('')

if False:
    # Visualize boards with features that have been found
    for i in range(len(images)):
        img = images[i]
        aruco.drawAxis(img, camera_matrix, dist_coeffs, \
            rvecs[i], tvecs[i], params['board']['square_length'])
        img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



#
# Show comparison
#

print('Camera matrix used in model')
print(cam_matrix)
print('Camera matrix as calibration result')
print(camera_matrix)
print('Absolute deviation of camera matrices')
print(cam_matrix-camera_matrix)

print('Distortion coefficients used in model')
print(cam_dist)
print('Distortion coefficients as calibration result')
print(dist_coeffs)
print('')

errors = []
for t, tcalib in zip(cam_trafos, calib_trafos):
    print(t)
    print(tcalib.inverse())
    dt, dr = Trafo3d().distance(t * tcalib)
    errors.append((dt, dr))
    print('--------------------')
errors = np.array(errors)
print(errors)
print(f'All trafos: dt={np.mean(errors[:,0]):.1f}, dr={np.rad2deg(np.mean(errors[:,1])):.2f} deg')



#
# Visualize board and all trafos
#

vis = SceneVisualizer()
mesh = CharucoBoard((params['board']['squares'][0],
                     params['board']['squares'][1]),
            params['board']['square_length'])
vis.add_mesh(mesh)
for t, tcalib in zip(cam_trafos, calib_trafos):
    vis.add_coordinate_system(size=100.0, T=t)
vis.show()