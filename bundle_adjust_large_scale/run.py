import copy
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load_multiple
from camsimlib.camera_model import CameraModel
from common.circle_detect import detect_circle_hough



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, '2d_calibrate_extrinsics')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load images
    images = image_load_multiple(os.path.join(data_dir, f'cam??.png'))
    num_views = len(images)

    # Load cameras
    cams = []
    for i in range(num_views):
        filename = os.path.join(data_dir, f'cam{i:02d}.json')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cams.append(cam)

    # Detect circle centers
    image = images[0]
    circles = detect_circle_hough(image, True)
