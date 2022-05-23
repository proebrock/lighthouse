import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import os
import sys
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.o3d_utils import mesh_generate_cs, mesh_generate_charuco_board
from camsimlib.camera_model import CameraModel



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



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    #
    # Read parameters that were used to generate calibration images;
    # this is the ground truth to compare the calibration results with
    #
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/2d_calibrate_single'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    aruco_dict = None
    aruco_board = None
    parameters = aruco.DetectorParameters_create()
    cam_matrix = None
    cam_dist = None
    cam_trafos = []

    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    for fname in filenames:
        with open(fname) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cam_trafos.append(cam.get_pose())
        if aruco_dict is None:
            # Assumption is that all images show the same aruco board
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            aruco_board = aruco.CharucoBoard_create( \
                params['board']['squares'][0], params['board']['squares'][1],
                params['board']['square_length'], params['board']['square_length'] / 2.0,
                aruco_dict)
            cam_matrix = cam.get_camera_matrix()
            cam_dist = cam.get_distortion()



    #
    # Run calibration
    #
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images_used, reprojection_error, calib_trafos, camera_matrix, dist_coeffs = \
        aruco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
    for index in np.where(~images_used)[0]:
        del cam_trafos[index]

    dc = np.zeros(12)
    dc[0:dist_coeffs.size] = dist_coeffs[0,:]
    dist_coeffs = dc



    #
    # Show comparison
    #
    print(f'Reprojection error: {reprojection_error:.2f} pixels')
    print('')
    with np.printoptions(precision=1, suppress=True):
        print('Camera matrix used in model')
        print(cam_matrix)
        print('Camera matrix as calibration result')
        print(camera_matrix)
        print('Deviation of camera matrices')
        print(camera_matrix - cam_matrix)
        print('')
    with np.printoptions(precision=3, suppress=True):
        print('Distortion coefficients used in model')
        print(cam_dist)
        print('Distortion coefficients as calibration result')
        print(dist_coeffs)
        print('Deviation of distortion coefficients')
        print(cam_dist - dist_coeffs)
        print('')

    errors = []
    for i, (t, tcalib) in enumerate(zip(cam_trafos, calib_trafos)):
        print(f'Index {i}')
        print(t)
        print(tcalib.inverse())
        dt, dr = t.distance(tcalib.inverse())
        errors.append((dt, np.rad2deg(dr)))
        print('--------------------')
    errors = np.array(errors)
    print(errors)
    print(f'All trafos: dt={np.mean(errors[:,0]):.1f}, dr={np.mean(errors[:,1]):.2f} deg')



    #
    # Visualize board and all trafos
    #
    board = mesh_generate_charuco_board((params['board']['squares'][0],
                                         params['board']['squares'][1]),
                params['board']['square_length'])
    scene = [ board ]
    for t in cam_trafos:
        scene.append(mesh_generate_cs(t, size=100.0))
    o3d.visualization.draw_geometries(scene)
