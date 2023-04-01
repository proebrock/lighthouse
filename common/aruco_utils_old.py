import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import cv2
import cv2.aruco as aruco

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d



def charuco_find_corners(filenames, aruco_dict, aruco_board):
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
            image_size = (gray.shape[1], gray.shape[0])
        else:
            assert image_size == (gray.shape[1], gray.shape[0])
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



def charuco_calibrate(filenames, aruco_dict, aruco_board, verbose=False):
    # Find corners in all images
    images, images_used, all_corners, all_ids, image_size = \
        charuco_find_corners(filenames, aruco_dict, aruco_board)
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
            # This function has moved in more recent OpenCV versions, see below
            #aruco.drawAxis(img, camera_matrix, dist_coeffs, \
            #    rvecs[i], tvecs[i], aruco_board.getSquareLength())
            cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, \
                rvecs[i], tvecs[i], aruco_board.getSquareLength())
            img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
            cv2.imshow(f'image{i:02d}', img)
            key = cv2.waitKey(0) & 0xff
            cv2.destroyAllWindows()
            if key == ord('q'):
                break
    if verbose:
        # Check if image points of all images do cover the whole area of the sensor
        # of the camera; this is premise for a good identification of the distortion
        # of the camera!
        fig, ax = plt.subplots()
        for corners in all_corners:
            ax.plot(corners[:, 0, 0], corners[:, 0, 1], 'o')
        ax.set_xlim((0, image_size[0]))
        ax.set_ylim((0, image_size[1]))
        ax.set_title('Distribution of image points for all images')
        plt.show()
    return images_used, reprojection_error, calib_trafos, camera_matrix, dist_coeffs
