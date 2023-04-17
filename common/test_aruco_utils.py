import sys
import os
import pytest
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel



def test_estimate_pose_valid():
    # Prepare scene
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    screen = board.generate_screen() # Screen CS == World CS
    screen_mesh = screen.get_mesh()
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam.scale_resolution(40)
    cam.place((130, 270, -300))
    cam.look_at((50, 50, 0))
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
    # Use camera and image to reconstruct the camera pose
    cam_to_board = board.estimate_pose(image, cam, verbose=False)
    cam_to_world = cam.get_pose().inverse() # Board CS == World CS
    # Check if sufficient
    dt, dr = cam_to_board.distance(cam_to_world)
    assert dt             < 1.0 # mm
    assert np.rad2deg(dr) < 0.1 # deg

