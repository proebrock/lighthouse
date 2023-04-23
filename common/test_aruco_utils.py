import sys
import os
import cv2
import pytest
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import CharucoBoard, MultiMarker
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from common.image_utils import image_show_multiple



def test_charuco_save_load_save_dict():
    # Generate board and save to dict
    ids = np.arange(17) + 17
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=ids)
    param_dict = {}
    board.dict_save(param_dict)
    # Generate second board, load, save and compare dicts
    board2 = CharucoBoard()
    board2.dict_load(param_dict)
    param_dict2 = {}
    board2.dict_save(param_dict2)
    assert param_dict == param_dict2



def test_charuco_estimate_pose_empty_image():
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    image = np.zeros((900, 1200, 3), dtype=np.uint8) # Empty black image
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    with pytest.raises(Exception) as ex:
        board.estimate_pose(image, cam, verbose=False)



def test_charuco_estimate_pose_valid():
    # Prepare scene: CharucoBoard and Screen
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    screen = board.generate_screen()
    cam_to_board = Trafo3d(t=[30, 50, 420], rpy=np.deg2rad([-35, 12, -165]))
    screen.set_pose(cam_to_board)
    screen_mesh = screen.get_mesh()
    # Prepare scene: CameraModel, camera CS is WORLD
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam.scale_resolution(30)
    # Visualize
    if False:
        screen_cs = screen.get_cs(size=100)
        cam_cs = cam.get_cs(size=100)
        cam_frustum = cam.get_frustum(size=200)
        o3d.visualization.draw_geometries([screen_cs, screen_mesh, cam_cs, cam_frustum])
    # Snap scene
    _, image, _ = cam.snap(screen_mesh)
    # Set background color for invalid pixels
    mask = np.all(np.isfinite(image), axis=2)
    image[~mask] = (0, 1, 1)
    image = (255.0 * image).astype(np.uint8)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()
    # Use camera and image to reconstruct the board pose
    cam_to_board_estim = board.estimate_pose(image, cam, verbose=False)
    dt, dr = cam_to_board.distance(cam_to_board_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg



def generate_board_poses(num_poses):
    rng = np.random.default_rng(0)
    translations = np.empty((num_poses, 3))
    translations[:,0] = rng.uniform(-100, 100, num_poses) # X
    translations[:,1] = rng.uniform(-100, 100, num_poses) # Y
    translations[:,2] = rng.uniform(-200, 200, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = rng.uniform(-20, 20, num_poses) # X
    rotations_rpy[:,1] = rng.uniform(-20, 20, num_poses) # Y
    rotations_rpy[:,2] = rng.uniform(-20, 20, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



def test_charuco_calibrate_camera():
    # Prepare scene: CharucoBoard and Screen
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    screen = board.generate_screen()
    # Prepare scene: CameraModel: Looks orthogonally in the middle of board
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    half_board_size = board.get_size_mm() / 2.0
    cam.place((half_board_size[0], half_board_size[1], -500))
    cam.look_at((half_board_size[0], half_board_size[1], 0))
    cam.roll(np.deg2rad(90))
    cam.scale_resolution(30)
    # Visualize
    if False:
        screen_cs = screen.get_cs(size=100)
        cam_cs = cam.get_cs(size=100)
        cam_frustum = cam.get_frustum(size=200)
        o3d.visualization.draw_geometries([screen_cs, screen_mesh, cam_cs, cam_frustum])
    num_images = 12
    world_to_screens = generate_board_poses(num_images)
    chip_size = cam.get_chip_size()
    images = np.zeros((num_images, chip_size[1], chip_size[0], 3), dtype=np.uint8)
    for i in range(num_images):
        print(f'Snapping image {i+1}/{num_images} ...')
        screen.set_pose(world_to_screens[i])
        screen_mesh = screen.get_mesh()
        _, image, _ = cam.snap(screen_mesh)
        # Set background color for invalid pixels
        mask = np.all(np.isfinite(image), axis=2)
        image[~mask] = (0, 1, 1)
        images[i, :, :, :] = (255.0 * image).astype(np.uint8)
    if False:
        image_show_multiple(images, single_window=True)
        plt.show()
    cam_recalib = CameraModel()
    # Identify simple camera model
    flags = cv2.CALIB_ZERO_TANGENT_DIST | \
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    cam_recalib, cam_to_boards_estim, reprojection_error = board.calibrate( \
        images, flags=flags, verbose=False)
    assert reprojection_error < 1.0
    # Check intrinsics
    d = np.abs(cam_recalib.get_chip_size() - cam.get_chip_size())
    assert np.all(d == 0)
    d = np.abs(cam_recalib.get_focal_length() - cam.get_focal_length())
    assert np.all(d / cam.get_chip_size())
    d = np.abs(cam_recalib.get_principal_point() - cam.get_principal_point())
    assert np.all(d / cam.get_principal_point())
    # Check trafos
    for i in range(num_images):
        cam_to_board = cam.get_pose().inverse() * world_to_screens[i]
        dt, dr = cam_to_board.distance(cam_to_boards_estim[i])
        assert dt             < 3.0 # mm
        assert np.rad2deg(dr) < 0.3 # deg



def test_charuco_estimate_two_poses_valid():
    # Prepare scene: first CharucoBoard and Screen
    board0 = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=np.arange(17))
    screen0 = board0.generate_screen()
    cam_to_board0 = Trafo3d(t=[-50, 100, 520], rpy=np.deg2rad([-35, 12, -165]))
    screen0.set_pose(cam_to_board0)
    screen_mesh0 = screen0.get_mesh()
    # Prepare scene: second CharucoBoard and Screen
    board1 = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=np.arange(17)+17)
    screen1 = board1.generate_screen()
    cam_to_board1 = Trafo3d(t=[100, 0, 500], rpy=np.deg2rad([15, -12, -185]))
    screen1.set_pose(cam_to_board1)
    screen_mesh1 = screen1.get_mesh()
    # Prepare scene: CameraModel, camera CS is WORLD
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam.scale_resolution(30)
    # Visualize
    if False:
        screen_cs0 = screen0.get_cs(size=100)
        screen_cs1 = screen1.get_cs(size=100)
        cam_cs = cam.get_cs(size=100)
        cam_frustum = cam.get_frustum(size=200)
        o3d.visualization.draw_geometries([screen_cs0, screen_mesh0, \
            screen_cs1, screen_mesh1, cam_cs, cam_frustum])
    # Snap scene
    _, image, _ = cam.snap([screen_mesh0, screen_mesh1])
    # Set background color for invalid pixels
    mask = np.all(np.isfinite(image), axis=2)
    image[~mask] = (0, 1, 1)
    image = (255.0 * image).astype(np.uint8)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()
    # Use camera and image to reconstruct the first board pose
    cam_to_board0_estim = board0.estimate_pose(image, cam, verbose=False)
    dt, dr = cam_to_board0.distance(cam_to_board0_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg
    # Use camera and image to reconstruct the first board pose
    cam_to_board1_estim = board1.estimate_pose(image, cam, verbose=False)
    dt, dr = cam_to_board1.distance(cam_to_board1_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg



def test_multimarker_save_load_save_dict():
    # Generate board and save to dict
    markers = MultiMarker(length_pix=80, length_mm=20.0,
        pose=Trafo3d(t=(10, -20, 30), rpy=np.deg2rad((120, -35, 215))))
    markers.add_marker(11, Trafo3d(t=(1, 2, 3)))
    markers.add_marker(21, Trafo3d(rpy=np.deg2rad((11, -22, 33))))
    param_dict = {}
    markers.dict_save(param_dict)
    # Generate second board, load, save and compare dicts
    markers2 = MultiMarker()
    markers2.dict_load(param_dict)
    param_dict2 = {}
    markers2.dict_save(param_dict2)
    assert param_dict == param_dict2



def test_multimarker_estimate_pose():
    # Prepare scene: multi-marker object
    markers = MultiMarker(length_pix=80, length_mm=20.0)
    d = 50
    markers.add_marker(11, Trafo3d(t=(-d, -d, 0)))
    markers.add_marker(12, Trafo3d(t=( d, -d, 0)))
    markers.add_marker(13, Trafo3d(t=(-d,  d, 0)))
    markers.add_marker(14, Trafo3d(t=( d,  d, 0)))
    meshes = markers.generate_meshes()
    # Prepare scene: cam0
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam0.scale_resolution(30)
    cam0.place((100, 0, -300))
    cam0.look_at((-10, 0, 0))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam1.scale_resolution(30)
    cam1.place((-50, 80, -250))
    cam1.look_at((0, 10, 0))
    cams = [ cam0, cam1 ]
    # Visualization
    if False:
        objects = [ \
            o3d.geometry.TriangleMesh.create_coordinate_frame(size=50),
            cam0.get_cs(size=100),
            cam0.get_frustum(size=200),
            cam1.get_cs(size=100),
            cam1.get_frustum(size=200),
        ]
        o3d.visualization.draw_geometries(objects + meshes)
    # Snap images
    cams = [ cam0, cam1 ]
    images = []
    for cam in cams:
        _, image, _ = cam.snap(meshes)
        # Set background color for invalid pixels
        mask = np.all(np.isfinite(image), axis=2)
        image[~mask] = (0, 1, 1)
        image = (255.0 * image).astype(np.uint8)
        images.append(image)
    # Visualization
    if False:
        for image in images:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(image)
        plt.show()
    # Estimate poses
    markers.estimate_pose(cams, images)
