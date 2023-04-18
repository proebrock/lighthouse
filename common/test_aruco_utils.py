import sys
import os
import pytest
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import CharucoBoard
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def test_save_load_save_dict():
    # Generate board and save to dict
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    param_dict = {}
    board.dict_save(param_dict)
    # Generate second board, load, save and compare dicts
    board2 = CharucoBoard()
    board2.dict_load(param_dict)
    param_dict2 = {}
    board2.dict_save(param_dict2)
    assert param_dict == param_dict2



def test_estimate_pose_empty_image():
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    with pytest.raises(Exception) as ex:
        board.estimate_pose(image, cam, verbose=False)



def test_estimate_pose_valid():
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
    image[~mask] = (0.5, 0.5, 0.5)
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



def test_estimate_two_poses_valid():
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
    image[~mask] = (0.5, 0.5, 0.5)
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

