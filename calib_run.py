import cv2
import cv2.aruco as aruco
import glob
import json
import numpy as np
from trafolib.trafo3d import Trafo3d



#
# Read parameters that were used to generate calibration images;
# this is the ground truth to compare the calibration results with
#

aruco_dict = None
board = None
parameters = aruco.DetectorParameters_create()
cam_matrix = None
cam_dist = None
cam_trafos = []

for fname in sorted(glob.glob('*.json')):
    with open(fname) as f:
        params = json.load(f)
    T = Trafo3d(t=params['cam']['trafo']['t'], q=params['cam']['trafo']['q'])
    cam_trafos.append(T)
    if aruco_dict is None:
        # Assumption is that all images show the same aruco board
        aruco_dict = aruco.Dictionary_get(params['board']['aruco_dict_index'])
        board = aruco.CharucoBoard_create( \
            params['board']['squares'][0], params['board']['squares'][1],
            params['board']['square_length'], params['board']['marker_length'],
            aruco_dict)
        cam_matrix = np.array([
            [ params['cam']['f'][0], 0.0, params['cam']['c'][0] ],
            [ 0.0, params['cam']['f'][1], params['cam']['c'][1] ],
            [ 0.0, 0.0, 1.0 ] ])
        cam_dist = params['cam']['distortion']


#
# Run the OpenCV calibration on the calibration images
#

allCorners = []
allIds = []

imageSize = None
images = []

for fname in sorted(glob.glob('*.png')):
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
print('')

print('Distortion coefficients used in model')
print(cam_dist)
print('Distortion coefficients as calibration result')
print(dist_coeffs)
print('')

for t, tcalib in zip(cam_trafos, calib_trafos):
    print(t)
    print(tcalib.inverse())
    print('--------------------')

