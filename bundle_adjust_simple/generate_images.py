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
from camsimlib.o3d_utils import mesh_generate_plane, save_shot



def generate_cameras(cam_scale=1.0):
    # cameras
    cameras = []
    # cam 0
    cam0 = CameraModel(chip_size=(40, 30), focal_length=40)
    cam0.place((410, 400, 707))
    cam0.look_at((-40, 50, 10))
    cam0.roll(np.deg2rad(-15))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=45)
    cam1.place((400, -420, 710))
    cam1.look_at((50, 40, -30))
    cam1.roll(np.deg2rad(3))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(40, 30), focal_length=43)
    cam2.place((-400, 350, 690))
    cam2.look_at((7, -9, 23))
    cam2.roll(np.deg2rad(-23))
    cameras.append(cam2)
    # cam 3
    cam3 = CameraModel(chip_size=(40, 30), focal_length=47)
    cam3.place((-430, -400, 650))
    cam3.look_at((30, 30, 30))
    cam3.roll(np.deg2rad(10))
    cameras.append(cam3)
    # Scale cameras
    for cam in cameras:
        cam.scale_resolution(cam_scale)
    return cameras



def visualize_scene(sphere, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, sphere ]
    for i, cam in enumerate(cameras):
        print(f'cam{i}: {cam.get_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Setup scene
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere_radius = 50.0
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    print('sphere bbox min', np.min(np.asarray(sphere.vertices), axis=0))
    print('sphere bbox max', np.max(np.asarray(sphere.vertices), axis=0))
    sphere_center = np.array((47,-61,-76))
    sphere.translate(sphere_center)
    sphere.paint_uniform_color((0.5, 0.2, 0.5))

    # Setup cameras
    cameras = generate_cameras(cam_scale=30.0)

    # Visualize
#    visualize_scene(sphere, cameras)

    for cam_no, cam in enumerate(cameras):
        basename = os.path.join(data_dir, f'cam{cam_no:02d}_image00')
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        depth_image, color_image, pcl = cam.snap(sphere)
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
        params['sphere'] = {}
        params['sphere']['center'] = sphere_center.tolist()
        params['sphere']['radius'] = sphere_radius
        with open(basename + '.json', 'w') as f:
           json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')

