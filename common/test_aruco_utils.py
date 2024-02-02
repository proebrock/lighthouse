import sys
import os
import cv2
import pytest
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.chessboard import Chessboard
from common.aruco_utils import CharucoBoard, MultiAruco
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from common.image_utils import image_3float_to_rgb, image_show_multiple



def test_charuco_chessboard_comparison():
    """ Make sure CharucoBoard and Chessboard are compatible and behave the same
    """
    charuco_board = CharucoBoard(squares=(5, 6), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    chess_board = Chessboard(squares=(5, 6), square_length_pix=80,
        square_length_mm=20.0)
    # Compare different sizes
    assert np.allclose(charuco_board.get_size_pix(),
        chess_board.get_size_pix())
    assert np.allclose(charuco_board.get_size_mm(),
        chess_board.get_size_mm())
    assert np.allclose(charuco_board.get_pixelsize_mm(),
        chess_board.get_pixelsize_mm())
    assert np.allclose(charuco_board.get_resolution_dpi(),
        chess_board.get_resolution_dpi())
    # Compare images
    charuco_image = charuco_board.generate_image()
    chess_image = chess_board.generate_image()
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(charuco_image)
        ax.set_title('CharucoBoard')
        ax = fig.add_subplot(122)
        ax.imshow(chess_image)
        ax.set_title('Chessboard')
        plt.show()
    assert charuco_image.shape == chess_image.shape
    assert charuco_image.dtype == chess_image.dtype
    # Screen
    charuco_screen = charuco_board.generate_screen()
    chess_screen = chess_board.generate_screen()
    assert np.allclose(charuco_screen.get_dimensions(),
        chess_screen.get_dimensions())
    assert charuco_screen.get_image().shape == chess_screen.get_image().shape
    # Mesh
    charuco_mesh = charuco_board.generate_mesh()
    charuco_aabb = charuco_mesh.get_axis_aligned_bounding_box()
    chess_mesh = chess_board.generate_mesh()
    chess_aabb = chess_mesh.get_axis_aligned_bounding_box()
    assert np.allclose(charuco_aabb.min_bound, chess_aabb.min_bound)
    assert np.allclose(charuco_aabb.max_bound, chess_aabb.max_bound)



def test_charuco_save_load_save_dict():
    # Generate board and save to dict
    ids = np.arange(17) + 17
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=ids,
        pose=Trafo3d(t=(10, -20, 30), rpy=np.deg2rad((120, -35, 215))))
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
        board.estimate_pose([ cam ], [ image ])



def test_charuco_estimate_pose():
    """ Test: Single calibrated camera, single CharucoBoard, single image;
    determine pose of board relative to camera
    """
    # Prepare scene: CharucoBoard and Screen
    cam_to_board = Trafo3d(t=[30, 50, 420], rpy=np.deg2rad([-35, 12, -165]))
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, pose=cam_to_board)
    screen = board.generate_screen()
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
    image = image_3float_to_rgb(image)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()
    # Use camera and image to reconstruct the board pose
    cam_to_board_estim, residuals_rms = board.estimate_pose([ cam ], [ image ])
    assert residuals_rms < 1.0
    dt, dr = cam_to_board.distance(cam_to_board_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg



def generate_board_poses(random_generator, num_poses):
    translations = np.empty((num_poses, 3))
    translations[:,0] = random_generator.uniform(-100, 100, num_poses) # X
    translations[:,1] = random_generator.uniform(-100, 100, num_poses) # Y
    translations[:,2] = random_generator.uniform(-200, 200, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = random_generator.uniform(-20, 20, num_poses) # X
    rotations_rpy[:,1] = random_generator.uniform(-20, 20, num_poses) # Y
    rotations_rpy[:,2] = random_generator.uniform(-20, 20, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



def test_charuco_calibrate_intrinsics(random_generator):
    """ Test: single uncalibrated camera, single CharucoBoard, multiple images;
    calibration of intrinsics of camera
    """
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
        screen_mesh = screen.get_mesh()
        cam_cs = cam.get_cs(size=100)
        cam_frustum = cam.get_frustum(size=200)
        o3d.visualization.draw_geometries([screen_cs, screen_mesh, cam_cs, cam_frustum])
    # Snap images
    num_images = 12
    world_to_screens = generate_board_poses(random_generator, num_images)
    chip_size = cam.get_chip_size()
    images = np.zeros((num_images, chip_size[1], chip_size[0], 3), dtype=np.uint8)
    for i in range(num_images):
        print(f'Snapping image {i+1}/{num_images} ...')
        screen.set_pose(world_to_screens[i])
        screen_mesh = screen.get_mesh()
        _, image, _ = cam.snap(screen_mesh)
        images[i, :, :, :] = image_3float_to_rgb(image)
    if False:
        image_show_multiple(images, single_window=True)
        plt.show()
    # Identify simple camera model
    flags = cv2.CALIB_ZERO_TANGENT_DIST | \
        cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
    cam_recalib, cam_to_boards_estim, reprojection_error = \
        board.calibrate_intrinsics(images, flags=flags)
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
        assert dt             < 20.0 # mm
        assert np.rad2deg(dr) < 2.0 # deg



def test_charuco_estimate_two_poses_valid():
    """ Test: single calibrated camera, two CharucoBoards with different IDs, one image:
    estimated poses of both boards relative to camera
    """
    # Prepare scene: first CharucoBoard and Screen
    cam_to_board0 = Trafo3d(t=[-50, 100, 520], rpy=np.deg2rad([-35, 12, -165]))
    board0 = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=np.arange(17),
        pose=cam_to_board0)
    screen0 = board0.generate_screen()
    screen_mesh0 = screen0.get_mesh()
    # Prepare scene: second CharucoBoard and Screen
    cam_to_board1 = Trafo3d(t=[100, 0, 500], rpy=np.deg2rad([15, -12, -185]))
    board1 = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0, ids=np.arange(17)+17,
        pose=cam_to_board1)
    screen1 = board1.generate_screen()
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
    image = image_3float_to_rgb(image)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()
    # Use camera and image to reconstruct the first board pose
    cam_to_board0_estim, residuals_rms = board0.estimate_pose([ cam ], [ image ])
    dt, dr = cam_to_board0.distance(cam_to_board0_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg
    # Use camera and image to reconstruct the second board pose
    cam_to_board1_estim, residuals_rms = board1.estimate_pose([ cam ], [ image ])
    dt, dr = cam_to_board1.distance(cam_to_board1_estim)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg



def test_multiaruco_save_load_save_dict():
    # Generate board and save to dict
    markers = MultiAruco(length_pix=80, length_mm=20.0,
        pose=Trafo3d(t=(10, -20, 30), rpy=np.deg2rad((120, -35, 215))))
    markers.add_marker(11, Trafo3d(t=(1, 2, 3)))
    markers.add_marker(21, Trafo3d(rpy=np.deg2rad((11, -22, 33))))
    param_dict = {}
    markers.dict_save(param_dict)
    # Generate second board, load, save and compare dicts
    markers2 = MultiAruco()
    markers2.dict_load(param_dict)
    param_dict2 = {}
    markers2.dict_save(param_dict2)
    assert param_dict == param_dict2



def test_multiaruco_estimate_pose_empty_image():
    markers = MultiAruco(length_pix=80, length_mm=20.0)
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    image = np.zeros((900, 1200, 3), dtype=np.uint8) # Empty black image
    with pytest.raises(Exception) as ex:
        markers.estimate_pose([ cam ], [ image ])



def test_multiaruco_estimate_pose():
    """ Test: Two calibrated cameras, one MultiAruco object, one image per camera;
    estimation of pose of MultiAruco object
    """
    # Prepare scene: multi-marker object
    w = 50.0
    markers = MultiAruco(length_pix=80, length_mm=w)
    d = 20.0
    markers.add_marker(0, Trafo3d(t=(-w-d, -w-d, 0)))
    markers.add_marker(1, Trafo3d(t=( d,   -w-d, 0)))
    markers.add_marker(2, Trafo3d(t=(-w-d, d, 0)))
    markers.add_marker(3, Trafo3d(t=( d,   d, 0)))
    if False:
        markers.plot2d()
        plt.show()
    # Prepare scene: cam0 and cam1
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam0.scale_resolution(30)
    cam0.place((100, 0, -300))
    cam0.look_at((-10, 0, 0))
    cam1 = CameraModel(chip_size=(32, 24), focal_length=(40, 40))
    cam1.scale_resolution(30)
    cam1.place((-50, 80, -200))
    cam1.look_at((0, 40, 0))
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
        objects.append(markers.generate_mesh())
        o3d.visualization.draw_geometries(objects)
    # Transform multi-marker object and cameras with same trafo
    world_to_center = Trafo3d(t=(-500, 200, -100), rpy=np.deg2rad((12, -127, 211)))
    markers.set_pose(world_to_center)
    for cam in cams:
        # world_to_cam = world_to_center * center_to_cam
        cam.set_pose(world_to_center * cam.get_pose())
    # Snap images
    mesh = markers.generate_mesh()
    images = []
    for cam in cams:
        _, image, _ = cam.snap(mesh)
        image = image_3float_to_rgb(image)
        images.append(image)
    # Visualization
    if False:
        for image in images:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(image)
        plt.show()
    # Estimate poses
    world_to_center_est, residuals_rms = markers.estimate_pose(cams, images)
    # Check results
    assert residuals_rms < 1.0
    dt, dr = world_to_center.distance(world_to_center_est)
    assert dt             < 3.0 # mm
    assert np.rad2deg(dr) < 0.3 # deg



def test_multiaruco_calibrate_extrinsics(random_generator):
    # Prepare scene: multi-marker object
    w = 40.0
    markers = MultiAruco(length_pix=80, length_mm=w)
    d = 20.0
    markers.add_marker(0, Trafo3d(t=(-w-d, -w-d, 0)))
    markers.add_marker(1, Trafo3d(t=( d,   -w-d, 0)))
    markers.add_marker(2, Trafo3d(t=(-w-d, d, 0)))
    markers.add_marker(3, Trafo3d(t=( d,   d, 0)))
    if False:
        markers.plot2d()
        plt.show()
    # Prepare scene: cam0 and cam1
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(55, 55))
    cam0.scale_resolution(30)
    cam0.place((100, -5, -600))
    cam0.look_at((-5, 0, 0))
    cam1 = CameraModel(chip_size=(32, 24), focal_length=(50, 50))
    cam1.scale_resolution(30)
    cam1.place((-110, 5, -650))
    cam1.look_at((0, 5, 0))
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
        objects.append(markers.generate_mesh())
        o3d.visualization.draw_geometries(objects)
    # Prepare images
    num_images = 4
    image_stacks = []
    for cam in cams:
        chip_size = cam.get_chip_size()
        images = np.zeros((num_images, chip_size[1], chip_size[0], 3), dtype=np.uint8)
        image_stacks.append(images)
    # Snap images
    world_to_screens = generate_board_poses(random_generator, num_images)
    for i in range(num_images):
        print(f'Snapping image {i+1}/{num_images} ...')
        mesh = markers.generate_mesh()
        mesh.transform(world_to_screens[i].get_homogeneous_matrix())
        for j, cam in enumerate(cams):
            _, image, _ = cam.snap(mesh)
            image_stacks[j][i, :, :, :] = image_3float_to_rgb(image)
    # Visualization
    if False:
        for images in image_stacks:
            image_show_multiple(images, single_window=True)
        plt.show()
    # Determine expected result: world_to_cams
    cam0_to_world = cams[0].get_pose().inverse()
    world_to_cams = []
    for cam in cams:
        T = cam0_to_world * cam.get_pose()
        world_to_cams.append(T)
    # Determine expected result: world_to_markers
    world_to_markers = []
    for i in range(num_images):
        T = cam0_to_world * world_to_screens[i]
        world_to_markers.append(T)
    # Estimate extrinsics
    world_to_cams_estim, world_to_markers_estim, residuals_rms = \
        markers.calibrate_extrinsics(cams, image_stacks)
    # Check results
    assert residuals_rms < 1.0
    for T, Test in zip(world_to_cams, world_to_cams_estim):
        dt, dr = T.distance(Test)
        assert dt             < 5.0 # mm
        assert np.rad2deg(dr) < 0.5 # deg
    for T, Test in zip(world_to_markers, world_to_markers_estim):
        dt, dr = T.distance(Test)
        assert dt             < 5.0 # mm
        assert np.rad2deg(dr) < 0.5 # deg
