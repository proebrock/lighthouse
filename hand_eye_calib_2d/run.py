import cv2
import cv2.aruco as aruco
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



#
# Run the OpenCV calibration on the calibration images
#
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
        dt, dr = base_to_board.distance(base_to_board_calculated)
        dtsum += dt
        drsum += dr
    return dtsum + 1.0 * np.rad2deg(drsum)

def hand_eye_calibrate_optim(base_to_flanges, cams_to_board):
    # Initial values
    base_to_board_0 = Trafo3d()
    flange_to_cam_0 = Trafo3d()
    x0 = toParam(base_to_board_0, flange_to_cam_0)

    options={ 'maxiter': 200000, 'maxfev': 200000, 'adaptive': True }
    tic = time.time()
    result = minimize(obj_fun, x0, args=(base_to_flanges, cams_to_board),
                      method='Nelder-Mead', options=options)
    toc = time.time()
    print(f'Optimization took {toc-tic:.1f}s')
    if result.success:
        base_to_board, flange_to_cam = fromParam(result.x)
        return base_to_board, flange_to_cam
    else:
        return None, None




if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    #
    # Read parameters that were used to generate calibration images;
    # this is the ground truth to compare the calibration results with
    #
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/leafstring/hand_eye_calib_2d'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

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
        cam_trafos.append(cam.get_camera_pose())
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
    # Run calibration
    #
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images_used, reprojection_error, cams_to_board, camera_matrix, dist_coeffs = \
        aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
    for index in np.where(~images_used)[0]:
        del base_to_flanges[index]
    assert(len(cams_to_board) == len(base_to_flanges))



    #
    # Calculating the hand-eye calibration
    #
    # Transformations involved:
    #
    # base_to_flanges * flange_to_cam * cams_to_board = base_to_board
    # (multiple, from   (single,        (multiple,      (single,
    # robot prg)        unknown)        from calib)     unknown by product)
    #
    base_to_board_estim, flange_to_cam_estim = \
        hand_eye_calibrate_optim(base_to_flanges, cams_to_board)
    print(base_to_board_real)
    print(base_to_board_estim)
    print(flange_to_cam_real)
    print(flange_to_cam_estim)