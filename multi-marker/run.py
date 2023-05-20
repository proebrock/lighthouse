import cv2
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_load
from common.aruco_utils import MultiAruco
from camsimlib.camera_model import CameraModel



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'multi-marker')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load camera and other settings
    basename = os.path.join(data_dir, 'cam00_image00')
    image = image_load(basename + '.png')
    with open(basename + '.json', 'r') as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    markers = MultiAruco()
    markers.dict_load(params['markers'])

    # Estimate pose of object in image
    cam_to_object = markers.get_pose()
    cam_to_object_estim, residuals_rms = markers.estimate_pose([cam], [image])
    print(f'residuals_rms:\n    {residuals_rms:.2f}')
    print(f'cam_to_object:\n    {cam_to_object}')
    print(f'cam_to_object estimated:\n    {cam_to_object_estim}')
    dt, dr = cam_to_object.distance(cam_to_object_estim)
    with np.printoptions(precision=2, suppress=True):
        print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
