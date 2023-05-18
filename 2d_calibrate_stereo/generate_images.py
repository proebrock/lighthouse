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
from common.aruco_utils import CharucoBoard
from common.image_utils import image_3float_to_rgb, image_save



def generate_cameras(cam_scale=1.0):
    # cameras
    cameras = []
    # Left camera
    cam_left = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        distortion=(0.1, -0.1))
    cam_left.set_pose(Trafo3d(t=(-120, 0, -1200)))
    cameras.append(cam_left)
    # Right camera
    cam_right = CameraModel(chip_size=(40, 30), focal_length=(50, 45),
        distortion=(-0.1, 0.1))
    cam_right.set_pose(Trafo3d(t=(120, 0, -1200)))
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



def visualize_scene(board, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    board_cs = board.get_cs(size=50.0)
    board_mesh = board.generate_mesh()
    objs = [ cs, board_cs, board_mesh ]
    for i, cam in enumerate(cameras):
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    board = CharucoBoard((6, 5), square_length_pix=75,
        square_length_mm=75.0, marker_length_mm=75.0/2)
    board_mesh = board.generate_mesh()
    board_pose = Trafo3d(t=-board_mesh.get_center()) # De-mean
    board.set_pose(board_pose)

    cameras = generate_cameras(cam_scale=30.0)

    #visualize_scene(board, cameras)

    i = 0
    while True:
        t = np.random.uniform(-100, 100, 3)
        rpy = np.deg2rad(np.random.uniform(-20, 20, 3))
        board.set_pose(board_pose * Trafo3d(t=t, rpy=rpy))
        board_mesh = board.generate_mesh()
        images = []
        for j, cam in enumerate(cameras):
            basename = os.path.join(data_dir, f'cam{j:02d}_image{i:02d}')
            print(f'Snapping image {basename} ...')
            # Snap image
            tic = time.monotonic()
            _, image, _ = cam.snap(board_mesh)
            toc = time.monotonic()
            print(f'    Snapping image took {(toc - tic):.1f}s')
            image = image_3float_to_rgb(image)
            # If board not fully visible, re-generate images
            if not board.all_points_visible(image):
                print('    # Not enough corners visible.')
                break
            images.append(image)
        if len(images) != len(cameras):
            continue
        for j, cam in enumerate(cameras):
            basename = os.path.join(data_dir, f'cam{j:02d}_image{i:02d}')
            # Save generated snap
            image_save(basename + '.png', images[j])
            # Save parameters
            params = {}
            params['cam'] = {}
            cam.dict_save(params['cam'])
            params['board'] = {}
            board.dict_save(params['board'])
            with open(basename + '.json', 'w') as f:
               json.dump(params, f, indent=4, sort_keys=True)

        i += 1
        if i == 12:
            break
    print('Done.')
