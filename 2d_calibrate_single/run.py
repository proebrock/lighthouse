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
from common.aruco_utils import charuco_calibrate



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
        charuco_calibrate(filenames, aruco_dict, aruco_board, verbose=False)
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
