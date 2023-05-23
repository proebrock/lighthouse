import json
import os
import sys
import time

import numpy as np
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import CharucoBoard
from common.image_utils import image_3float_to_rgb, image_save
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def generate_board_poses(num_poses):
    rng = np.random.default_rng(0)
    translations = np.empty((num_poses, 3))
    translations[:,0] = rng.uniform(-100, 100, num_poses) # X
    translations[:,1] = rng.uniform(-100, 100, num_poses) # Y
    translations[:,2] = rng.uniform(-200, 200, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = rng.uniform(-20, 20, num_poses) # X
    rotations_rpy[:,1] = rng.uniform(-20, 20, num_poses) # Y
    rotations_rpy[:,2] = rng.uniform(-20, 20, num_poses) # Z
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

    # Prepare scene: CharucoBoard and Screen
    board = CharucoBoard(squares=(5, 7), square_length_pix=80,
        square_length_mm=20.0, marker_length_mm=10.0)
    screen = board.generate_screen()

    # Prepare scene: CameraModel: Looks orthogonally in the middle of board
    cam = CameraModel(
        chip_size=(40, 30),
        focal_length=(50, 55),
        #distortion=(-0.1, 0.1, 0.05, -0.05, 0.2, 0.08),
    )
    half_board_size = board.get_size_mm() / 2.0
    cam.place((half_board_size[0], half_board_size[1], -600))
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
    world_to_screens = generate_board_poses(num_images)
    chip_size = cam.get_chip_size()
    for i in range(num_images):
        basename = os.path.join(data_dir, f'image{i:02d}')
        screen.set_pose(world_to_screens[i])
        screen_mesh = screen.get_mesh()
        # Snap scene
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        _, image, _ = cam.snap(screen_mesh)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        image = image_3float_to_rgb(image)
        image_save(basename + '.png', image)
        # Save parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        params['board'] = {}
        board.dict_save(params['board'])
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

    print('Done.')
