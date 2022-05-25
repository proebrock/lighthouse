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
    mesh_generate_cs, mesh_generate_charuco_board, save_shot



def generate_cameras(cam_scale=1.0):
    # cameras
    cameras = []
    # cam 0
    cam0 = CameraModel(chip_size=(40, 40), focal_length=(35, 35))
    cam0.place((400, 400, 700))
    cam0.look_at((100, 150, 0))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(80, 85))
    cam1.place((500, -400, 1800))
    cam1.look_at((200, 100, 0))
    cam1.roll(np.deg2rad(60))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(50, 40), focal_length=(65, 60))
    cam2.place((-450, 500, 800))
    cam2.look_at((100, 100, 0))
    cam2.roll(np.deg2rad(-20))
    cameras.append(cam2)
    # cam 3
    cam3 = CameraModel(chip_size=(30, 30), focal_length=(30, 25))
    cam3.place((-300, -300, 600))
    cam3.look_at((200, 200, 0))
    cam3.roll(np.deg2rad(-60))
    cameras.append(cam3)
    # Scale cameras
    for cam in cameras:
        cam.scale_resolution(cam_scale)
    return cameras



def visualize_scene(board_pose, board, cameras):
    print(f'board: {board_pose}')
    cs = mesh_generate_cs(board_pose, size=100.0)
    current_board = copy.deepcopy(board)
    mesh_transform(current_board, board_pose)
    objs = [ cs, current_board ]
    for i, cam in enumerate(cameras):
        print(f'cam{i}: {cam.get_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



def generate_board_poses(num_poses):
    translations = np.empty((num_poses, 3))
    translations[:,0] = np.random.uniform(-100, 100, num_poses) # X
    translations[:,1] = np.random.uniform(-100, 100, num_poses) # Y
    translations[:,2] = np.random.uniform(-200, 200, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = np.random.uniform(-20, 20, num_poses) # X
    rotations_rpy[:,1] = np.random.uniform(-20, 20, num_poses) # Y
    rotations_rpy[:,2] = np.random.uniform(-20, 20, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    squares = (6, 5)
    square_length = 75.0
    board = mesh_generate_charuco_board(squares, square_length)
    board_pose = Trafo3d()

    cameras = generate_cameras(cam_scale=30.0)
    if True:
        # Place cam0 in origin
        board_pose = cameras[0].get_pose().inverse()
        for cam in cameras:
            cam.set_pose(board_pose * cam.get_pose())
#    visualize_scene(board_pose, board, cameras)

    board_poses = generate_board_poses(12)
    for i, pose in enumerate(board_poses):
        current_board = copy.deepcopy(board)
        current_board_pose = board_pose * pose
        mesh_transform(current_board, current_board_pose)
        for j, cam in enumerate(cameras):
            basename = os.path.join(data_dir, f'cam{j:02d}_image{i:02d}')
            print(f'Snapping image {basename} ...')
            # Snap image
            tic = time.monotonic()
            depth_image, color_image, pcl = cam.snap(current_board)
            toc = time.monotonic()
            print(f'    Snapping image took {(toc - tic):.1f}s')
            # Save generated snap
            # Save PCL in camera coodinate system, not in world coordinate system
            pcl.transform(cam.get_pose().inverse().get_homogeneous_matrix())
            save_shot(basename, depth_image, color_image, pcl)
            # Save all image parameters
            params = {}
            params['cam'] = {}
            cam.dict_save(params['cam'])
            params['board'] = {}
            params['board']['squares'] = squares
            params['board']['square_length'] = square_length
            params['board']['pose'] = {}
            params['board']['pose']['t'] = current_board_pose.get_translation().tolist()
            params['board']['pose']['q'] = current_board_pose.get_rotation_quaternion().tolist()
            with open(basename + '.json', 'w') as f:
               json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')
