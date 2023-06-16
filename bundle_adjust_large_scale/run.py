import copy
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_load_multiple
from camsimlib.camera_model import CameraModel
from common.circle_detect import detect_circle_hough
from color_matcher import match_colors
from common.bundle_adjust import bundle_adjust_points_and_poses
from common.registration import estimate_transform



def visualize_scene(cams, P, cams_estimated=None, P_estimated=None):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ cs ]
    for cam in cams:
        objects.append(cam.get_cs(size=50))
        objects.append(cam.get_frustum(size=200, color=(0, 0, 1)))
    for i in range(P.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.paint_uniform_color((0, 0, 1))
        sphere.translate(P[i, :])
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        objects.append(sphere)
    if cams_estimated is not None:
        for cam in cams_estimated:
            objects.append(cam.get_cs(size=20))
            objects.append(cam.get_frustum(size=120, color=(1, 0, 0)))
    if P_estimated is not None:
        for i in range(P_estimated.shape[0]):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
            sphere.paint_uniform_color((1, 0, 0))
            sphere.translate(P_estimated[i, :])
            sphere.compute_vertex_normals()
            sphere.compute_triangle_normals()
            objects.append(sphere)
    o3d.visualization.draw_geometries(objects)



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

    # Load points and colors
    filename = os.path.join(data_dir, f'cam00.json')
    with open(filename) as f:
        params = json.load(f)
        P = np.asarray(params['sphere_centers'])
        model_colors = np.asarray(params['sphere_colors'])
    num_points = P.shape[0]

    # Load cameras
    cams = []
    poses = []
    for i in range(num_views):
        filename = os.path.join(data_dir, f'cam{i:02d}.json')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cams.append(cam)
        poses.append(cam.get_pose())

    # Assemble on-chip point matrix for bundle adjustment:
    # Project 3D points to chips; just for checking results, normally
    # we have no access to real 3D positions in P
    p = np.zeros((num_points, num_views, 2))
    for i, cam in enumerate(cams):
        p[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distances

    # Assemble on-chip point matrix for bundle adjustment:
    # Use circle centers in provided images
    p_reconstructed = np.zeros((num_points, num_views, 2))
    p_reconstructed[:] = np.NaN
    for i in range(num_views):
        image = images[i]
        # Detect circles in image
        circles = detect_circle_hough(image, min_center_distance=40, min_radius=10,
            max_radius=100, verbose=False)
        # Sample colors from center of circle
        circle_centers_int = np.round(circles[:, 0:2]).astype(int)
        colors = image[circle_centers_int[:, 1], circle_centers_int[:, 0], :]
        colors = colors / 255.0
        # Match colors to original colors
        indices = match_colors(colors, model_colors, verbose=False)
        p_reconstructed[indices, i, :] = circles[:, 0:2]

    if False:
        # With detecting circle centers and with matching of colors we make some
        # errors; this compares the observations reconstructed from the images
        # with the ground truth;
        # These errors make bundle adjustment more complex due to the presence
        # of outliers.
        circle_detect_errors = p_reconstructed - p
        circle_detect_errors = np.sqrt(np.sum(np.square(circle_detect_errors), axis=2))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plot = ax.matshow(circle_detect_errors)
        fig.colorbar(plot)
        ax.set_title('Circle detection errors')

    # Inital estimates: This is crucial to calculate a successful bundle adjustment
    P_init = np.random.uniform(-200, 200, (num_points, 3))
    pose_init = num_views * [ Trafo3d(t=(0, 0, -1000)) ]

    # Run bundle adjustment: due to outliers in circle detection we use stable
    # optimization with loss function
    P_estimated, poses_estimated, residuals = bundle_adjust_points_and_poses( \
        cam, p_reconstructed, P_init=P, pose_init=poses, full=True,
        optimizer_opt={ 'loss': 'soft_l1' })

    # Result from bundle adjustment may be translated and/or rotated and/or scaled
    # compared to the original point P and poses; we estimate a transformation
    # between P and P_estimated and use this to compensate for this
    if True:
        if True:
            # Compensate for translation, rotation and scaling
            groundtruth_to_estimated, scale = estimate_transform(P, P_estimated, estimate_scale=True)
        else:
            # Compensate for translation, rotation and NOT for scaling
            groundtruth_to_estimated = estimate_transform(P, P_estimated)
            scale = 1.0
        P_estimated = groundtruth_to_estimated * (scale * P_estimated)
        for i in range(num_views):
            t = groundtruth_to_estimated * (scale * poses_estimated[i].get_translation())
            rot = groundtruth_to_estimated.get_rotation_matrix() @ poses_estimated[i].get_rotation_matrix()
            poses_estimated[i] = Trafo3d(t=t, mat=rot)

    # Calculate point errors
    point_errors = np.sqrt(np.sum(np.square(P_estimated - P), axis=1))

    # Calculate pose errors
    pose_errors_trans = []
    pose_errors_rot = []
    for pose, estimated_pose in zip(poses, poses_estimated):
        dt, dr = pose.distance(estimated_pose)
        pose_errors_trans.append(dt)
        pose_errors_rot.append(dr)
    pose_errors_trans = np.asarray(pose_errors_trans)
    pose_errors_rot = np.asarray(pose_errors_rot)
    pose_errors_rot = np.rad2deg(pose_errors_rot)

    # Point reconstruction errors plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(point_errors)
    ax.grid()
    ax.set_title('Point reconstruction errors')
    ax.set_xlabel('Point index')
    ax.set_ylabel('Error (mm)')

    # Pose reconstruction errors plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(pose_errors_trans, color='b')
    ax2.plot(pose_errors_rot, color='r')
    ax.set_title(f'Pose reconstruction errors')
    ax.set_xlabel('Pose index')
    ax.set_ylabel('Translational error (mm)', color='b')
    ax2.set_ylabel('Rotational error (deg)', color='r')

    # Residual errors plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = ax.matshow(residuals)
    fig.colorbar(plot)
    ax.set_title('Residual errors')

    plt.show()

    # Visualize reconstructed scene
    cams_estimated = []
    for T in poses_estimated:
        c = copy.deepcopy(cam)
        c.set_pose(T)
        cams_estimated.append(c)
    visualize_scene(cams, P, cams_estimated, P_estimated)

