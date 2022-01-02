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
    images_color = []
    cam_poses = []
    cam_matrices = []
    cam_dists = []
    for cidx in range(2):
        basename = os.path.join(data_dir, f'{title}_cam{cidx:02d}')
        # Load images
        img_color = cv2.imread(basename + '_color.png')
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        images.append(img_gray)
        images_color.append(img_color)
        # Load camera parameters
        with open(os.path.join(basename + '.json'), 'r') as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cam_poses.append(cam.get_camera_pose())
        cam_matrices.append(cam.get_camera_matrix())
        cam_dists.append(cam.get_distortion())
    cam_r_to_cam_l = cam_poses[1].inverse() * cam_poses[0]
    return images, images_color, cam_r_to_cam_l, cam_matrices, cam_dists



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
    data_dir = 'b'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/stereo_vision'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load scene data
    scene_titles = ( 'ideal', 'realistic')
    images, images_color, cam_r_to_cam_l, cam_matrices, cam_dists = load_scene(data_dir, scene_titles[1])
    image_size = (images[0].shape[1], images[0].shape[0])
    E, F = calculate_stereo_matrices(cam_r_to_cam_l, cam_matrices[0], cam_matrices[1])
    if False:
        # Show input images
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
    image_fixed_l = cv2.remap(images[0], mapx[0], mapy[0],
        cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    image_fixed_r = cv2.remap(images[1], mapx[1], mapy[1],
        cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    images_fixed = [ image_fixed_l, image_fixed_r ]
    if True:
        # Show rectified images
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images_fixed[0], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Rectified left')
        ax = fig.add_subplot(122)
        ax.imshow(images_fixed[1], cmap='gray')
        ax.set_axis_off()
        ax.set_title('Rectified right')

    if True:
        # Save rectified images in order to use them with stereo_matcher_gui
        cv2.imwrite('image_fixed_l.png', image_fixed_l)
        cv2.imwrite('image_fixed_r.png', image_fixed_r)

    if True:
        # Show same row from rectified left and right images
        row_index = 400
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(image_fixed_l[row_index, :], 'r', label='left')
        ax.plot(image_fixed_r[row_index, :], 'g', label='right')
        ax.legend(loc='best', fancybox=True, framealpha=0.5)
        ax.set_xlabel('Image column')
        ax.set_ylabel('Pixel brightness')
        ax.set_title(f'Row #{row_index}')
        ax.grid()



    # Make some calculations to properly configure the stereo matcher
    baseline = np.linalg.norm(cam_r_to_cam_l.get_translation())
    print(f'Base line: {baseline:.0f} mm')
    distance_search_range = np.array((400, 1400))
    distance_cutoff = 1300 # Points beyond that distance are filtered, must be lower than distance_search_range[1]
    print(f'Z-Distance search range: {distance_search_range} mm')
    focal_length = (cam_matrices[0][0,0] + cam_matrices[1][0,0]) / 2.0
    print(f'Focal length: {focal_length}')
    estim_disparity_range = (baseline * focal_length) / distance_search_range
    estim_disparity_range = np.sort(estim_disparity_range)
    with np.printoptions(precision=0, suppress=True):
        print(f'Estimated disparity range: {estim_disparity_range} pix')
    estim_disparity_range = 16.0 * np.round(estim_disparity_range / 16.0) # round to muliples of 16
    estim_disparity_range = estim_disparity_range.astype(int)
    print(f'Estimated disparity range (rounded): {estim_disparity_range} pix')

    # Run stereo block matching to get disparity
    stereo_matcher = cv2.StereoBM_create()
    stereo_matcher.setMinDisparity(estim_disparity_range[0])
    stereo_matcher.setNumDisparities(estim_disparity_range[1] - estim_disparity_range[0])
    stereo_matcher.setBlockSize(15)
    image_disparity = stereo_matcher.compute(images_fixed[0], images_fixed[1])
    image_disparity = image_disparity.astype(np.float32) / 16.0

    # Calculate distance from disparity
    image_distance = (baseline * focal_length) / image_disparity
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image_distance, cmap='viridis')
        ax.set_axis_off()
        ax.set_title('Distance')

    # Calculate point clouds from disparity
    points = cv2.reprojectImageTo3D(image_disparity, disp_to_depth_map)
    points = np.reshape(points, (-1, 3))
    colors = images_color[0].reshape((-1, 3)) / 255.0
    # Filter points by distance
    mask_valid = image_distance.ravel() <= distance_cutoff
    points = points[mask_valid, :]
    colors = colors[mask_valid, :]

    # End of 2d visualizations
    plt.show()

    if True:
        # Show final result as a colored point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        o3d.visualization.draw_geometries([cs, pcl])


    """
    TODO:
    * Draw epilines, links:
        https://www.reddit.com/r/computervision/comments/g6lwiz/poor_quality_stereo_matching_with_opencv/
        https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv
        https://docs.opencv.org/4.5.5/d9/d0c/group__calib3d.html#ga19e3401c94c44b47c229be6e51d158b7
    * Compare with ground truth
    * Writing documentation for 2d_calibrate_stereo and stereo_vision
    """

