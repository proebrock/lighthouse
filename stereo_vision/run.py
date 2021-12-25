import copy
import cv2
import json
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.close('all')

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d



def load_scene(data_dir, title):
    images = []
    cam_poses = []
    cam_matrices = []
    for cidx in range(2):
        basename = os.path.join(data_dir, f'{title}_cam{cidx:02d}')
        # Load images
        img = cv2.imread(basename + '_color.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)
        # Load camera parameters
        with open(os.path.join(basename + '.json'), 'r') as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cam_poses.append(cam.get_camera_pose())
        cam_matrices.append(cam.get_camera_matrix())
    cam_pose_l2r = cam_poses[0].inverse() * cam_poses[1]
    return images, cam_pose_l2r, cam_matrices



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/stereo_vision'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load scene data
    images, cam_pose_l2r, cam_matrices = load_scene(data_dir, 'ideal')

    # Calculate essential and fundamental matrices
    # TODO: Compare implementation with that of cv::stereoCalibrate; matrix scaling?
    t = cam_pose_l2r.get_translation()
    R = cam_pose_l2r.get_rotation_matrix()
    S = np.array([
        [ 0, -t[2], t[1] ],
        [ t[2], 0, -t[0] ],
        [ -t[1], t[0], 0 ],
    ])
    E = S @ R
    print(f'Essential matrix:\n{E}')
    M_r = cam_matrices[1]
    M_l = cam_matrices[0]
    F = np.linalg.inv(M_r).T @ E @ np.linalg.inv(M_l)
    print(f'Fundamental matrix:\n{F}')