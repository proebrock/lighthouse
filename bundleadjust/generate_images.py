import numpy as np
import open3d as o3d
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, \
    mesh_generate_charuco_board, show_images, save_shot



np.random.seed(42) # Random but reproducible



def generate_scene():
    # board
    squares = (6, 5)
    square_length = 50.0
    board = mesh_generate_charuco_board(squares, square_length)
    board.translate(-board.get_center()) # de-mean
    # cameras
    cameras = []
    # cam 0
    cam0 = CameraModel(chip_size=(40, 40), focal_length=(35, 35))
    cam0.place_camera((0, 0, 600))
    cam0.look_at((50, -50, 0))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(100, 100))
    cam1.place_camera((500, 500, 1200))
    cam1.look_at((100, 0, 0))
    cam1.roll_camera(np.deg2rad(60))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(50, 40), focal_length=(60, 60))
    cam2.place_camera((-400, 100, 800))
    cam2.look_at((0, -100, 0))
    cam2.roll_camera(np.deg2rad(-20))
    cameras.append(cam2)
    # cam 3
    cam3 = CameraModel(chip_size=(30, 30), focal_length=(30, 25))
    cam3.place_camera((0, -300, 500))
    cam3.look_at((0, -50, 0))
    cam3.roll_camera(np.deg2rad(-120))
    cameras.append(cam3)
    return board, cameras



def visualize_scene(board, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, board ]
    for cam in cameras:
        print(cam.get_camera_pose())
        objs.append(cam.get_cs(size=100.0))
        #objs.append(cam.get_frustum(size=100.0))
    o3d.visualization.draw_geometries(objs)



board, cameras = generate_scene()
visualize_scene(board, cameras)

#depth_image, color_image, pcl = cam3.snap(board)
#show_images(depth_image, color_image)
