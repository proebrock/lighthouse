import copy
import matplotlib.pyplot as plt
import json
import numpy as np
import os
import sys
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import MultiAruco
from common.image_utils import image_load_multiple
from camsimlib.camera_model import CameraModel



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'hand_eye_calib_2d')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load cameras
    num_cams = 4
    cams = []
    world_to_cams_real = []
    for i in range(num_cams):
        filename = os.path.join(data_dir, f'cam{i:02d}_image00.json')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cams.append(cam)
        world_to_cams_real.append(cam.get_pose())
    # Transform camera calibrations relative to camera 0
    T = copy.deepcopy(world_to_cams_real[0]).inverse()
    for i in range(len(world_to_cams_real)):
        world_to_cams_real[i] = T * world_to_cams_real[i]

    # Load images
    image_stacks = []
    for i in range(num_cams):
        pattern = os.path.join(data_dir, f'cam{i:02d}_image??.png')
        images = image_load_multiple(pattern)
        image_stacks.append(images)

    # Load cube
    filename = os.path.join(data_dir, f'cam00_image00.json')
    with open(filename) as f:
        params = json.load(f)
    cube = MultiAruco()
    cube.dict_load(params['cube'])

    print('Running extrinsics calibration ...')
    world_to_cams, world_to_markers, residuals_rms = \
        cube.calibrate_extrinsics(cams, image_stacks)
    print(f'    Done, residual RMS is {residuals_rms:.2f}')

    print('Comparing results ...')
    for i, (wc, wcr) in enumerate(zip(world_to_cams, world_to_cams_real)):
        dt, dr = wcr.distance(wc)
        print(f'    Errors for cam{i}: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
