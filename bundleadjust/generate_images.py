import copy
import json
import numpy as np
import open3d as o3d
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, \
    mesh_generate_charuco_board, show_images, save_shot



def generate_cameras(cam_scale=1.0):
    # cameras
    cameras = []
    # cam 0
    cam0 = CameraModel(chip_size=(40, 40), focal_length=(35, 35))
    cam0.place_camera((0, 0, 600))
    cam0.look_at((200, 50, 0))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(100, 100))
    cam1.place_camera((500, 500, 1200))
    cam1.look_at((250, 100, 0))
    cam1.roll_camera(np.deg2rad(60))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(50, 40), focal_length=(60, 60))
    cam2.place_camera((-400, 100, 800))
    cam2.look_at((150, 0, 0))
    cam2.roll_camera(np.deg2rad(-20))
    cameras.append(cam2)
    # cam 3
    cam3 = CameraModel(chip_size=(30, 30), focal_length=(30, 25))
    cam3.place_camera((50, -300, 500))
    cam3.look_at((150, 50, 0))
    cam3.roll_camera(np.deg2rad(-60))
    cameras.append(cam3)
    # Scale cameras
    for cam in cameras:
        cam.scale_resolution(cam_scale)
    return cameras



def visualize_scene(board, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, board ]
    for i, cam in enumerate(cameras):
        print(f'cam{i}: {cam.get_camera_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



def generate_board_poses(num_poses):
    translations = np.empty((num_poses, 3))
    translations[:,0] = np.random.uniform(-50, 50, num_poses) # X
    translations[:,1] = np.random.uniform(-50, 50, num_poses) # Y
    translations[:,2] = np.random.uniform(-100, 100, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = np.random.uniform(-10, 10, num_poses) # X
    rotations_rpy[:,1] = np.random.uniform(-10, 10, num_poses) # Y
    rotations_rpy[:,2] = np.random.uniform(-10, 10, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



np.random.seed(42) # Random but reproducible
data_dir = 'a'
if not os.path.exists(data_dir):
    raise Exception('Target directory does not exist.')

squares = (6, 5)
square_length = 50.0
board = mesh_generate_charuco_board(squares, square_length)
board_pose = Trafo3d()

cameras = generate_cameras(cam_scale=20.0)
if True:
    # Place cam0 in origin
    board_pose = cameras[0].get_camera_pose().inverse()
    mesh_transform(board, board_pose)
    for cam in cameras:
        cam.set_camera_pose(board_pose * cam.get_camera_pose())
#visualize_scene(board, cameras)

board_poses = generate_board_poses(12)
for i, pose in enumerate(board_poses):
    current_board = copy.deepcopy(board)
    mesh_transform(current_board, pose)
    for j, cam in enumerate(cameras):
        basename = os.path.join(data_dir, f'cam{j:02d}_image{i:02d}')
        print(f'Snapping image {basename} ...')
        # Snap image
        tic = time.process_time()
        depth_image, color_image, pcl = cam.snap(current_board)
        toc = time.process_time()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        save_shot(basename, depth_image, color_image, pcl)
        # Save all image parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        params['board'] = {}
        params['board']['squares'] = squares
        params['board']['square_length'] = square_length
        params['board']['pose'] = {}
        params['board']['pose']['t'] = board_pose.get_translation().tolist()
        params['board']['pose']['q'] = board_pose.get_rotation_quaternion().tolist()
        with open(basename + '.json', 'w') as f:
           json.dump(params, f, indent=4, sort_keys=True)
print('Done.')
