import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_load_multiple, \
    image_show_multiple
from common.pixel_matcher import ImageMatcher
from common.mesh_utils import mesh_load
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector



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

    # Load configuration
    filename = os.path.join(data_dir, 'mesh.ply')
    mesh = mesh_load(filename)
    filename = os.path.join(data_dir, 'projector.json')
    #projector = ShaderProjector()
    #projector.json_load(filename)
    cam_filenames = sorted(glob.glob(os.path.join(data_dir, 'cam??.json')))
    cams = []
    for i, filename in enumerate(cam_filenames):
        cam = CameraModel()
        cam.json_load(filename)
        cams.append(cam)
    filename = os.path.join(data_dir, 'matcher.json')
    matcher = ImageMatcher()
    matcher.json_load(filename)

    # Load images
    images = []
    for cam_no in range(len(cams)):
        filenames = os.path.join(data_dir, f'image????_cam{cam_no:04}.png')
        images.append(image_load_multiple(filenames))

    cam_no = 0
    print(f'Matching for camera {cam_no} ...')
    tic = time.monotonic()
    indices = matcher.match(images[cam_no])
    toc = time.monotonic()
    print(f'Matching image took {(toc - tic):.1f}s')
