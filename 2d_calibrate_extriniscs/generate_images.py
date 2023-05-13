import copy
import json
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from common.aruco_utils import MultiAruco
from common.image_utils import image_3float_to_rgb, image_save
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def generate_calibration_cube(width, border):
    cube = MultiAruco(length_pix=20, length_mm=width)
    cube.add_marker(1, Trafo3d(t=(-width/2, -width/2, -border-width/2),
        rpy=np.deg2rad((0, 0, 0))))
    cube.add_marker(2, Trafo3d(t=(border+width/2, -width/2, -width/2),
        rpy=np.deg2rad((0, -90, 0))))
    cube.add_marker(3, Trafo3d(t=(-width/2, -border-width/2, width/2),
        rpy=np.deg2rad((-90, 0, 0))))
    cube.add_marker(4, Trafo3d(t=(-width/2, border+width/2, -width/2),
        rpy=np.deg2rad((90, 0, 0))))
    cube.add_marker(5, Trafo3d(t=(-border-width/2, -width/2, width/2),
        rpy=np.deg2rad((0, 90, 0))))
    cube.add_marker(6, Trafo3d(t=(width/2, -width/2, border+width/2),
        rpy=np.deg2rad((0, 180, 0))))
    return cube



def generate_cameras():
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam0.place((100, 0, -100))
    cam0.look_at((10, 5, -5))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(50, 55))
    cam1.place((0, 100, 100))
    cam1.look_at((10, 5, -5))
    cam2 = CameraModel(chip_size=(40, 30), focal_length=(45, 45))
    cam2.place((-120, 140, 0))
    cam2.look_at((-10, -10, 0))
    cam3 = CameraModel(chip_size=(40, 30), focal_length=(45, 50))
    cam3.place((100, -120, 0))
    cam3.look_at((0, -10, -10))
    cams = [ cam0, cam1, cam2, cam3 ]
    for cam in cams:
        cam.scale_resolution(40)
    return  cams



def visualize_scene(cube, box, cams):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
    objects = [ cs, cube.get_cs(10), cube.generate_mesh(), box ]
    for cam in cams:
        objects.append(cam.get_cs(10))
        objects.append(cam.get_frustum(50))
    o3d.visualization.draw_geometries(objects)



def generate_poses(num_poses):
    rng = np.random.default_rng(0)
    translations = np.empty((num_poses, 3))
    translations[:,0] = rng.uniform(-30, 30, num_poses) # X
    translations[:,1] = rng.uniform(-30, 30, num_poses) # Y
    translations[:,2] = rng.uniform(-30, 30, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = rng.uniform(-90, 90, num_poses) # X
    rotations_rpy[:,1] = rng.uniform(-90, 90, num_poses) # Y
    rotations_rpy[:,2] = rng.uniform(-90, 90, num_poses) # Z
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

    width = 20
    border = 2
    cube = generate_calibration_cube(width, border)
    box = o3d.geometry.TriangleMesh.create_box()
    box.translate((-0.5, -0.5, -0.5))
    box.paint_uniform_color((1, 1, 1))
    box.scale(width + 2*border - 0.5, (0, 0, 0))

    cams = generate_cameras()

    cube_poses = generate_poses(10)

    #visualize_scene(cube, box, cams)

    for i, pose in enumerate(cube_poses):
        cube.set_pose(pose)
        mesh = o3d.geometry.TriangleMesh(box)
        mesh.transform(pose.get_homogeneous_matrix())
        mesh = mesh + cube.generate_mesh()
        for j, cam in enumerate(cams):
            basename = os.path.join(data_dir, f'cam{j:02d}_image{i:02d}')
            # Snap scene
            print(f'Snapping image {basename} ...')
            tic = time.monotonic()
            _, image, _ = cam.snap(mesh)
            toc = time.monotonic()
            print(f'    Snapping image took {(toc - tic):.1f}s')
            # Save generated snap
            image = image_3float_to_rgb(image)
            image_save(basename + '.png', image)
            # Save parameters
            params = {}
            params['cam'] = {}
            cam.dict_save(params['cam'])
            params['cube'] = {}
            cube.dict_save(params['cube'])
            with open(basename + '.json', 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)

    print('Done.')
