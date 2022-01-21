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
    pcls = []
    cams = []
    for cidx in range(2):
        basename = os.path.join(data_dir, f'{title}_cam{cidx:02d}')
        # Load images
        img_color = cv2.imread(basename + '_color.png')
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
        images.append(img_gray)
        images_color.append(img_color)
        # Load point cloud (ground truth)
        pcl = o3d.io.read_point_cloud(basename + '.ply')
        pcls.append(pcl)
        # Load camera parameters
        with open(os.path.join(basename + '.json'), 'r') as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cams.append(cam)
    return images, images_color, pcls, cams



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
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/stereo_vision'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load scene data
    scene_titles = ( 'ideal', 'distorted', \
        'displaced_tx', 'displaced_ty', 'displaced_tz', \
        'displaced_rx', 'displaced_ry', 'displaced_rz', \
        'displaced', 'distorted_displaced')
    images, images_color, pcls, cams = load_scene(data_dir, scene_titles[4])
    cam_r_to_cam_l = cams[1].get_camera_pose().inverse() * cams[0].get_camera_pose()
    image_size = (images[0].shape[1], images[0].shape[0])
    E, F = calculate_stereo_matrices(cam_r_to_cam_l,
        cams[0].get_camera_matrix(), cams[1].get_camera_matrix())
    if True:
        # Show input images
        row = 490
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images[0], cmap='gray')
        ax.axhline(row, color='r')
        ax.text(10, row-10, f'row={row}', color='r')
        ax.set_axis_off()
        ax.set_title('Original left')
        ax = fig.add_subplot(122)
        ax.imshow(images[1], cmap='gray')
        ax.axhline(row, color='r')
        ax.text(10, row-10, f'row={row}', color='r')
        ax.set_axis_off()
        ax.set_title('Original right')
    if True:
        # Show epiline example
        # Calculate epilines using fundamental matrix F
        points_left = np.array(((357, 146), (873, 298), (748, 490), (398, 636), (960, 636), (961, 830)))
        lines_right = cv2.computeCorrespondEpilines(points_left, whichImage=1, F=F)
        # Result for each input point: (a, b, c) with line equation ax+by+c=0
        lines_right = np.reshape(lines_right, (-1, 3))
        # Display
        colors = ('r', 'g', 'b', 'c', 'm', 'y')
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images[0], cmap='gray')
        for i, p in enumerate(points_left):
            ax.plot(p[0], p[1], 'o', color=colors[i])
        ax.set_axis_off()
        ax.set_title('Original left, points')
        ax = fig.add_subplot(122)
        ax.imshow(images[1], cmap='gray')
        for i, l in enumerate(lines_right):
            x = np.array((0, images[1].shape[0]))
            y = (-l[0] * x - l[2]) / l[1]
            ax.plot(x, y, '-', color=colors[i])
        ax.set_axis_off()
        ax.set_title('Original right, corresponding epilines')


    # Calculate rectification
    print('----------------------------')
    print('Input to cv2.stereoRectify:')
    with np.printoptions(precision=3, suppress=True):
        print(f'cameraMatrix1=\n{cams[0].get_camera_matrix()}')
        print(f'distCoeffs1={cams[0].get_distortion()}')
        print(f'cameraMatrix1=\n{cams[1].get_camera_matrix()}')
        print(f'distCoeffs1={cams[1].get_distortion()}')
        print(f'imageSize={image_size}')
        print(f'R={cam_r_to_cam_l.get_rotation_matrix()}')
        print(f'T={cam_r_to_cam_l.get_translation()}')
    flags = cv2.CALIB_ZERO_DISPARITY
    alpha = -1
    rect_l, rect_r, proj_l, proj_r, disp_to_depth_map, roi_l, roi_r = \
        cv2.stereoRectify( \
        cams[0].get_camera_matrix(), cams[0].get_distortion(), \
        cams[1].get_camera_matrix(), cams[1].get_distortion(), \
        image_size, cam_r_to_cam_l.get_rotation_matrix(), cam_r_to_cam_l.get_translation(), \
        None, None, None, None, None, flags=flags, alpha=alpha)
    print('----------------------------')
    print('Output of cv2.stereoRectify:')
    with np.printoptions(precision=3, suppress=True):
        print(f'R1=\n{rect_l}')
        print(f'R2=\n{rect_r}')
        print(f'P1=\n{proj_l}')
        print(f'P2=\n{proj_r}')
        print(f'Q={disp_to_depth_map}')
    print('----------------------------')
    # Output 3x3 rectification transform (rotation matrix) for the cameras.
    # This matrix brings points given in the unrectified camera's coordinate system
    # to points in the rectified camera's coordinate system. In more technical terms,
    # it performs a change of basis from the unrectified camera's coordinate system
    # to the rectified camera's coordinate system.
    rects = [ rect_l, rect_r ]
    # Output 3x4 projection matrix in the new (rectified) coordinate systems for the
    # cameras, i.e. it projects points given in the rectified camera coordinate system
    # into the rectified camera's image.
    projs = [ proj_l, proj_r ]
    # Optional output rectangles inside the rectified images where all the pixels are
    # valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely
    # to be smaller
    rois = [ roi_l, roi_r ]
    mapx_l, mapy_l = cv2.initUndistortRectifyMap( \
        cams[0].get_camera_matrix(), cams[0].get_distortion(),
        rects[0], projs[0], image_size, cv2.CV_32FC1)
    mapx_r, mapy_r = cv2.initUndistortRectifyMap( \
        cams[1].get_camera_matrix(), cams[1].get_distortion(),
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
        row = 490
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(images_fixed[0], cmap='gray')
        ax.axhline(row, color='r')
        ax.text(10, row-10, f'row={row}', color='r')
        ax.set_axis_off()
        ax.set_title('Rectified left')
        ax = fig.add_subplot(122)
        ax.imshow(images_fixed[1], cmap='gray')
        ax.axhline(row, color='r')
        ax.text(10, row-10, f'row={row}', color='r')
        ax.set_axis_off()
        ax.set_title('Rectified right')

    if True:
        # Save rectified images in order to use them with stereo_matcher gui
        cv2.imwrite('image_fixed_l.png', image_fixed_l)
        cv2.imwrite('image_fixed_r.png', image_fixed_r)

    if True:
        # Show same row from rectified left and right images
        row_index = 650
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
    focal_length = (cams[0].get_focal_length()[0] + cams[1].get_focal_length()[0]) / 2.0
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
    mask_valid = image_distance.ravel() <= distance_cutoff
    if True:
        # Show depth image
        samples = np.array((
            (300, 840),
            (180, 420),
            (680, 420),
            (680, 840),
            ))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image_distance, cmap='viridis')
        for r, c in samples:
            ax.plot(c, r, 'or')
            ax.text(c+30, r, f'z={image_distance[r, c]:.0f}mm')
        ax.set_axis_off()
        ax.set_title('Depth')


    # OpenCV productes strange Q matrix (disp_to_depth_map); "fixing" matrix generates more realistic results
#    pp = cams[0].get_principal_point()
#    disp_to_depth_map[0,3] = -pp[0]
#    disp_to_depth_map[1,3] = -pp[1]

    # Calculate point clouds from disparity image using OpenCV
    points = cv2.reprojectImageTo3D(image_disparity, disp_to_depth_map)
    points = np.reshape(points, (-1, 3))
    # Determine point cloud color and filter points and colors by validity of distance
    colors = images_color[0].reshape((-1, 3)) / 255.0
    points = points[mask_valid, :]
    colors = colors[mask_valid, :]

    # End of 2d visualizations, start of 3d visualizations
    plt.show()

    if True:
        # Show final result as a colored point cloud
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        pcl.colors = o3d.utility.Vector3dVector(colors)
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        o3d.visualization.draw_geometries([cs, pcl])

    if True:
        # Show ground truth vs. reconstructed point cloud
        pcl_nominal = copy.deepcopy(pcls[0])
        np.asarray(pcl_nominal.colors)[:] = (0, 1, 0)
        pcl_estimated = o3d.geometry.PointCloud()
        pcl_estimated.points = o3d.utility.Vector3dVector(points)
        col = np.zeros_like(colors)
        col[:] = (1, 0, 0)
        pcl_estimated.colors = o3d.utility.Vector3dVector(col)

        cam0_cs = cams[0].get_cs(size=50)
        cam0_frustum = cams[0].get_frustum(size=300)
        cam1_cs = cams[1].get_cs(size=50)
        cam1_frustum = cams[1].get_frustum(size=300)
        o3d.visualization.draw_geometries([pcl_nominal, pcl_estimated, \
            cam0_cs, cam0_frustum, cam1_cs, cam1_frustum])


    """
    TODO:
    * Failure to compare with ground truth for "realistic" version :-(((
    * Writing documentation for 2d_calibrate_stereo and stereo_vision
    """

