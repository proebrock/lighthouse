import copy
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import pytest

import open3d as o3d

from trafolib.trafo3d import Trafo3d
from common.bundle_adjust import bundle_adjust_points, bundle_adjust_points_and_poses
from common.registration import estimate_transform
from camsimlib.camera_model import CameraModel



def visualize_scene(cams, P, cams_estimated=None, P_estimated=None):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ cs ]
    for cam in cams:
        objects.append(cam.get_cs(size=50))
        objects.append(cam.get_frustum(size=200, color=(0, 0, 1)))
    for i in range(P.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
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
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
            sphere.paint_uniform_color((1, 0, 0))
            sphere.translate(P_estimated[i, :])
            sphere.compute_vertex_normals()
            sphere.compute_triangle_normals()
            objects.append(sphere)
    o3d.visualization.draw_geometries(objects)



def scene_to_chip(cams, P):
    points = np.zeros((P.shape[0], len(cams), 2))
    for i, cam in enumerate(cams):
        p = cam.scene_to_chip(P)
        p = p[:, 0:2] # Omit distances
        points[:, i, :] = p
    return points



def generate_visibility_mask(num_points, num_views):
    visibility_mask = np.ones((num_points, num_views), dtype=bool)
    num_rows_reduced = num_points // 5
    row_choices = np.random.choice(num_points, num_rows_reduced, replace=False)
    for r in row_choices:
        min_number_views = 2 # we need at least 2 cams...
        col_choices = np.random.choice(num_views, num_views - min_number_views, replace=True)
        visibility_mask[r, col_choices] = False
    #print(f'Visibility {np.sum(visibility_mask)}/{visibility_mask.size}')
    return visibility_mask



def test_bundle_adjust_points_variying_visibility():
    # Setup scene
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(200, 0 ,0)))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(-200, 0 ,0)))
    cam2 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(0, 200 ,0)))
    cam3 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(0, -200, 0)))
    cams = [ cam0, cam1, cam2, cam3 ]
    P = np.array((
        (-100, 200, 800),
        (100, 0, 800),
        (-100, 200, 600),
        (50, -50, 900),
        ))
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)

    # Disable some observations
    visible_mask = np.array((
        (False, False, False, False), # Point 0 visibile by 0 cameras
        (True,  False, False, False), # Point 1 visibile by 1 cameras
        (True,  False, True,  False), # Point 2 visibile by 2 cameras
        (True,  True,  True,  False), # Point 3 visibile by 3 cameras
    ), dtype=bool)                    # camera 3 does not see anything
    p[~visible_mask, :] = np.NaN

    # Check bundle adjust with varying visibilities; if not enough observations
    # available, we expect exceptions
    with pytest.raises(ValueError):
        bundle_adjust_points(cams, p[0:4, :, :])
    with pytest.raises(ValueError):
        bundle_adjust_points(cams, p[1:4, :, :])
    bundle_adjust_points(cams, p[3:4, :, :])
    bundle_adjust_points(cams, p[2:4, :, :], full=True)



def test_bundle_adjust_points_residuals():
    # Setup scene
    cam0 = CameraModel(chip_size=(800, 600), focal_length=(800, 800),
        pose=Trafo3d(t=(250, 0 ,0)))
    cam1 = CameraModel(chip_size=(800, 600), focal_length=(800, 800),
        pose=Trafo3d(t=(-250, 0 ,0)))
    cams = [ cam0, cam1 ]
    P = np.array(((0, 0, 1000),))
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)
    # Introduce error of d pixels in opposite directions
    d = 10.0
    p[0, 0, 1] -= d
    p[0, 1, 1] += d
    # We expect this error to be present in the result of the bundle adjustment
    expected_residuals = [ d, d ]
    expected_distances = [ 12.5, 12.5 ]

    # Run bundle adjustment
    P_estimated, residuals, distances = bundle_adjust_points(cams, p, full=True)

    # Check results
    assert np.max(np.abs((P_estimated - P))) < 1e-2
    assert np.max(np.abs((expected_residuals - residuals))) < 1e-2
    assert np.max(np.abs((expected_distances - distances))) < 1e-2



def test_bundle_adjust_points_upscaled():
    # Setup scene
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(200, 0 ,0)))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(-200, 0 ,0)))
    cam2 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(0, 200 ,0)))
    cam3 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(0, -200, 0)))
    cams = [ cam0, cam1, cam2, cam3 ]
    num_points = 5000
    P = np.zeros((num_points, 3))
    P[:, 0] = np.random.uniform(-500, 500, num_points)
    P[:, 1] = np.random.uniform(-500, 500, num_points)
    P[:, 2] = np.random.uniform(500, 1500, num_points)
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)

    # Disable some observations
    visibility_mask = generate_visibility_mask(num_points, len(cams))
    p[~visibility_mask, :] = np.NaN

    # Run bundle adjustment
    P_estimated, residuals, distances = bundle_adjust_points(cams, p, full=True)

    # Check results
    absdiff = np.abs((P_estimated - P))
    assert np.max(absdiff) < 0.1

    assert np.max(residuals[visibility_mask]) < 0.1
    assert np.all(np.isnan(residuals[~visibility_mask]))

    assert np.max(distances[visibility_mask]) < 0.1
    assert np.all(np.isnan(distances[~visibility_mask]))



def test_bundle_adjust_points_and_poses_basic():
    # Generate camera model
    cam = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam.scale_resolution(20)

    if False:
        # Random camera placements

        # Generate 3D points
        num_points = 20
        P = np.random.uniform(-200, 200, (num_points, 3))
        num_views = 40
        world_to_cam_1 = Trafo3d(t=(0, 0, -1000))
        poses = []
        for i in range(num_views):
            # Small movements in point coordinate system
            t = np.random.uniform(-50, 50, 3)
            rpy = np.random.uniform(-180, 180, 3)
            # Transformed into camera movement
            points_trafo = Trafo3d(t=t, rpy=np.deg2rad(rpy))
            world_to_cam_n = points_trafo * world_to_cam_1
            poses.append(world_to_cam_n)
        # Create individual camera per view
        cams = []
        for T in poses:
            c = copy.deepcopy(cam)
            c.set_pose(T)
            cams.append(c)
    else:
        # Spiral-shaped continuous camera movement

        # Generate 3D points
        num_points = 40
        P = np.random.uniform(-200, 200, (num_points, 3))
        num_views = 40
        height_max = 1000
        height_min = 200
        radius = 500
        angles = np.linspace(0, 2*np.pi, num_views + 1)[0:-1]
        cams = []
        poses = []
        for i, angle in enumerate(angles):
            c = copy.deepcopy(cam)
            c.place((
                radius * np.cos(angle),
                radius * np.sin(angle),
                height_min + ((height_max - height_min) * i) / num_views
                ))
            c.look_at((0, 0, 0))
            cams.append(c)
            poses.append(c.get_pose())

    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)

    # Disable some observations
    visibility_mask = generate_visibility_mask(num_points, len(cams))
    p[~visibility_mask, :] = np.NaN

    # Inital estimates: This is crucial to calculate a successful bundle adjustment
    P_init = np.random.uniform(-200, 200, (num_points, 3))
    pose_init = num_views * [ Trafo3d(t=(0, 0, -1000)) ]

    # Run bundle adjustment
    P_estimated, poses_estimated, residuals = bundle_adjust_points_and_poses( \
        cam, p, P_init=P_init, pose_init=pose_init, full=True)

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

    if False:
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
        fig.colorbar(plot, label='Error (pixels)')
        ax.set_xlabel('Image/view index')
        ax.set_ylabel('3D point index')
        ax.set_title('Residual errors')

        plt.show()

        # Visualize reconstructed scene
        cams_estimated = []
        for T in poses_estimated:
            c = copy.deepcopy(cam)
            c.set_pose(T)
            cams_estimated.append(c)
        visualize_scene(cams, P, cams_estimated, P_estimated)

    # Check for results
    assert np.max(point_errors) < 1e-6
    assert np.max(pose_errors_trans) < 1e-6
    assert np.max(pose_errors_rot) < 1e-6
    assert np.max(residuals[visibility_mask]) < 1e-6
    assert np.all(np.isnan(residuals[~visibility_mask]))
