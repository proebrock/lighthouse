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
    cam_dists = []
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
        cam_dists.append(cam.get_distortion())
    cam_r_to_cam_l = cam_poses[1].inverse() * cam_poses[0]
    return images, cam_r_to_cam_l, cam_matrices, cam_dists



def calculate_stereo_matrices(cam_r_to_cam_l, camera_matrix_l, camera_matrix_r):
    t = cam_r_to_cam_l.get_translation()
    R = cam_r_to_cam_l.get_rotation_matrix()
    # Essential matrix E
    S = np.array([
        [ 0, -t[2], t[1] ],
        [ t[2], 0, -t[0] ],
        [ -t[1], t[0], 0 ],
    ])
    E = S @ R
    # Fundamental matrix F
    F = np.linalg.inv(camera_matrix_r).T @ E @ np.linalg.inv(camera_matrix_l)
    if not np.isclose(F[2, 2], 0.0):
        F = F / F[2, 2]
    return E, F



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/stereo_vision'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load scene data
    images, cam_r_to_cam_l, cam_matrices, cam_dists = load_scene(data_dir, 'realistic')
    image_size = images[0].shape
    E, F = calculate_stereo_matrices(cam_r_to_cam_l, cam_matrices[0], cam_matrices[1])
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images[0], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Original left')
        ax = fig.add_subplot(122)
        ax.imshow(images[1], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Original right')

    # Calculate rectification
    rect_l, rect_r, proj_l, proj_r, disp_to_depth_map, roi_l, roi_r = \
        cv2.stereoRectify(cam_matrices[0], cam_dists[0], cam_matrices[1], cam_dists[1], \
        image_size, cam_r_to_cam_l.get_rotation_matrix(), cam_r_to_cam_l.get_translation(), \
        None, None, None, None, None, cv2.CALIB_ZERO_DISPARITY, -1)
    rects = [ rect_l, rect_r ]
    projs = [ proj_l, proj_r ]
    rois = [ roi_l, roi_r ]
    mapx_l, mapy_l = cv2.initUndistortRectifyMap(cam_matrices[0], cam_dists[0], \
        rects[0], projs[0], image_size, cv2.CV_32FC1)
    mapx_r, mapy_r = cv2.initUndistortRectifyMap(cam_matrices[1], cam_dists[1], \
        rects[1], projs[1], image_size, cv2.CV_32FC1)
    mapx = [ mapx_l, mapx_r ]
    mapy = [ mapy_l, mapy_r ]

    # Rectify images
    image_fixed_l = cv2.remap(images[0], mapx[0], mapy[0], cv2.INTER_NEAREST)
    image_fixed_r = cv2.remap(images[1], mapx[1], mapy[1], cv2.INTER_NEAREST)
    images_fixed = [ image_fixed_l, image_fixed_r ]
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images_fixed[0], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Rectified left')
        ax = fig.add_subplot(122)
        ax.imshow(images_fixed[1], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Rectified right')

    # Calculate depth image
    stereo_matcher = cv2.StereoBM_create()
#    stereo_matcher.setMinDisparity(4)
#    stereo_matcher.setNumDisparities(128)
#    stereo_matcher.setBlockSize(21)
#    stereo_matcher.setSpeckleRange(16)
#    stereo_matcher.setSpeckleWindowSize(45)
    image_depth = stereo_matcher.compute(images_fixed[0], images_fixed[1])
    print(np.min(image_depth), np.max(image_depth))
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image_depth.astype(np.float64) / 1024.0, cmap='viridis')
        ax.set_axis_off()
        ax.set_title('Depth')

    """
    TODO:
    * Properly calculate depth map
    * Calculate point cloud
    * Understand coordinate system of result of calculations
    * Compare with ground truth
    """

    plt.show()
