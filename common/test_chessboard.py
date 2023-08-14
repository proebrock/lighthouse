import sys
import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2



sys.path.append(os.path.abspath('../'))
from common.chessboard import Chessboard
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_3float_to_rgb, image_show_multiple
from camsimlib.camera_model import CameraModel



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



def test_calibrate_intrinsics(random_generator):
    """ Test: single uncalibrated camera, single CharucoBoard, multiple images;
    calibration of intrinsics of camera
    """
    # Prepare scene: CharucoBoard and Screen
    board = Chessboard(squares=(5, 6), square_length_pix=80,
        square_length_mm=20.0)
    #board.plot2d()
    screen = board.generate_screen()
    # Prepare scene: CameraModel: Looks orthogonally in the middle of board
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    half_board_size = board.get_size_mm() / 2.0
    cam.place((half_board_size[0], half_board_size[1], -700))
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
    cam_recalib, reprojection_error = \
        board.calibrate_intrinsics(images, flags=flags)
    assert reprojection_error < 1.0
    # Check intrinsics
    d = np.abs(cam_recalib.get_chip_size() - cam.get_chip_size())
    assert np.all(d == 0)
    d = np.abs(cam_recalib.get_focal_length() - cam.get_focal_length())
    assert np.all(d / cam.get_chip_size())
    d = np.abs(cam_recalib.get_principal_point() - cam.get_principal_point())
    assert np.all(d / cam.get_principal_point())
