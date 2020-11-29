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



def generate_scene(cam0_origin=True):
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
    cam3.place_camera((50, -300, 500))
    cam3.look_at((0, -50, 0))
    cam3.roll_camera(np.deg2rad(-60))
    cameras.append(cam3)
    # Put first camera coordinate system in origin
    if cam0_origin:
        trafo = cam0.get_camera_pose().inverse()
        mesh_transform(board, trafo)
        for cam in cameras:
            cam.set_camera_pose(trafo * cam.get_camera_pose())
    return board, cameras



def visualize_scene(board, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, board ]
    for i, cam in enumerate(cameras):
        print(f'cam{i}: {cam.get_camera_pose()}')
        objs.append(cam.get_cs(size=100.0))
        #objs.append(cam.get_frustum(size=100.0))
    o3d.visualization.draw_geometries(objs)



def generate_board_poses(num_poses):
    translations = np.empty((num_poses, 3))
    translations[:,0] = np.random.uniform(-100, 100, num_poses) # X
    translations[:,1] = np.random.uniform(-100, 100, num_poses) # Y
    translations[:,2] = np.random.uniform(-200, 200, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = np.random.uniform(-10, 10, num_poses) # X
    rotations_rpy[:,1] = np.random.uniform(-10, 10, num_poses) # Y
    rotations_rpy[:,2] = np.random.uniform(-10, 10, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



board, cameras = generate_scene()
#visualize_scene(board, cameras)

board_poses = generate_board_poses(10)

#depth_image, color_image, pcl = cam3.snap(board)
#show_images(depth_image, color_image)
