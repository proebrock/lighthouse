import copy
import json
import numpy as np
import open3d as o3d
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_generate_charuco_board, save_shot
from trafolib.trafo3d import Trafo3d



def generate_robot_moves():
    trans = np.array([
        [266.43,1.09,562.76],
        [456.25,185.46,562.76],
        [198.86,185.46,562.73],
        [508.30,-219.83,562.73],
        [255.88,-436.00,562.66],
        [513.36,290.89,338.67],
        [348.26,-4.87,211.91],
        [252.44,236.20,343.54],
        [252.43,236.20,343.54],
        [252.43,236.21,196.25],
        [500.36,160.68,262.30],
        [495.50,-54.13,220.77],
        [495.51,-212.41,205.86],
        [293.71,-347.32,205.85],
        [218.40,-334.02,191.62],
        [322.13,13.48,343.96],
        [429.57,-69.31,343.99],
        [395.66,62.03,344.00],
        [347.98,141.33,724.79],
        [534.67,-2.11,362.71],
        [534.67,-170.86,525.62]])
    rotq = np.array([
        [0.501857,0.49545,0.503242,0.499416],
        [0.502037,0.483699,0.645345,0.312288],
        [0.64617,0.26198,0.482599,0.530027],
        [0.381282,0.668482,0.418086,0.48266],
        [0.321661,0.443981,0.283501,0.786793],
        [0.555717,0.375997,0.703099,0.235493],
        [0.488867,0.528082,0.484466,0.497424],
        [0.64533,0.254113,0.578142,0.4298],
        [0.149527,-0.129807,0.873857,0.444033],
        [0.19222,-0.179736,0.922463,0.282503],
        [0.203864,0.366846,0.826879,0.374346],
        [0.353028,0.598183,0.644794,0.319044],
        [0.225941,0.825205,0.403053,0.324863],
        [0.318943,0.875851,0.0420352,0.359712],
        [0.188174,0.940838,-0.167741,0.226446],
        [0.501853,0.495384,0.503219,0.499509],
        [0.00383812,-0.0164639,-0.700543,-0.71341],
        [0.709592,0.704432,0.0152532,-0.00476847],
        [0.497374,0.465054,0.598136,0.422583],
        [0.377941,0.589328,0.619921,0.354332],
        [0.197923,0.470766,0.619772,0.595894]])
    return list((Trafo3d(t=t, q=q) for t, q in zip(trans, rotq)))



def visualize_scene(board, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, board ]
    for cam in cameras:
#        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Calibration board
    baseTboard = Trafo3d(t=(530, 180, 0),  rpy=np.deg2rad((0, 0, 180)))
    squares = (5, 7)
    square_length = 50.0
    board = mesh_generate_charuco_board(squares, square_length)
    board.transform(baseTboard.get_homogeneous_matrix())

    # Camera positions
    baseTflanges = generate_robot_moves()
    flangeTcam = Trafo3d(t=(20, -20,  6), rpy=np.deg2rad((110, -87, -20)))
    cam = CameraModel(chip_size=(4056, 3040),
                      focal_length=4028,
                      principal_point=(1952, 1559),
                      distortion=(-0.5, 0.3, 0, 0, -0.11))
    cameras = []
    for baseTflange in baseTflanges:
        c = copy.deepcopy(cam)
#        c.scale_resolution(0.1) # Scale camera resolution
        c.set_camera_pose(baseTflange * flangeTcam)
        cameras.append(c)

#    visualize_scene(board, cameras)

    for step, cam in enumerate(cameras):
        # Snap scene
        basename = os.path.join(data_dir, f'cam00_image{step:02d}')
#        print(f'Snapping image {basename} ...')
#        tic = time.process_time()
#        depth_image, color_image, pcl = cam.snap(board)
#        toc = time.process_time()
#        print(f'    Snapping image took {(toc - tic):.1f}s')
#        # Save generated snap
#        # Save PCL in camera coodinate system, not in world coordinate system
#        pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
#        save_shot(basename, depth_image, color_image, pcl)
        # Save all image parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        params['board'] = {}
        params['board']['squares'] = squares
        params['board']['square_length'] = square_length
        params['base_to_board'] = {}
        params['base_to_board']['t'] = baseTboard.get_translation().tolist()
        params['base_to_board']['q'] = baseTboard.get_rotation_quaternion().tolist()
        params['base_to_flange'] = {}
        params['base_to_flange']['t'] = baseTflanges[step].get_translation().tolist()
        params['base_to_flange']['q'] = baseTflanges[step].get_rotation_quaternion().tolist()
        params['flange_to_cam'] = {}
        params['flange_to_cam']['t'] = flangeTcam.get_translation().tolist()
        params['flange_to_cam']['q'] = flangeTcam.get_rotation_quaternion().tolist()
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')