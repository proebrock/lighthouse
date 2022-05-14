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
    # Left camera
    cam_left = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        distortion=(0.1, -0.1))
    cam_left.set_pose(Trafo3d(t=(-120, 0, 1200), rpy=np.deg2rad((180, 0, 0))))
    cameras.append(cam_left)
    # Right camera
    cam_right = CameraModel(chip_size=(40, 30), focal_length=(50, 45),
        distortion=(-0.1, 0.1))
    cam_right.set_pose(Trafo3d(t=(120, 0, 1200), rpy=np.deg2rad((180, 0, 0))))
    if True:
        # Realistic scenario
        T = cam_right.get_pose()
        T = T * Trafo3d(t=(7, 3, -14), rpy=np.deg2rad((-1.5, 3, 2)))
        cam_right.set_pose(T)
    cameras.append(cam_right)
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
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    squares = (6, 5)
    square_length = 75.0
    board = mesh_generate_charuco_board(squares, square_length)
    board_pose = Trafo3d(t=-board.get_center()) # De-mean

    cameras = generate_cameras(cam_scale=30.0)
    if False:
        # Place cam0 in origin
        board_pose = cameras[0].get_pose().inverse()
        for cam in cameras:
            cam.set_pose(board_pose * cam.get_pose())
    #visualize_scene(board_pose, board, cameras)

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
