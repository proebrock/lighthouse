import copy
import numpy as np
import open3d as o3d
import json
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d
from camsimlib.o3d_utils import mesh_transform, save_shot
from camsimlib.shader_point_light import ShaderPointLight



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Generate scenes
    mesh = o3d.io.read_triangle_mesh('../data/fox_head.ply')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh.scale(500, center=mesh.get_center())
    mesh_transform(mesh, Trafo3d(rpy=np.deg2rad([0, 180, 180])))
    poses = [
        Trafo3d(t=(0, 0, 1000)),
        Trafo3d(t=(0, 0, 2000)),
        Trafo3d(t=(0, 0, 3000)),
        ]
    scenes = []
    for p in poses:
        m = copy.deepcopy(mesh)
        mesh_transform(m, p)
        scenes.append(m)

    # Generate cameras
    tof_cam = CameraModel(chip_size=(64, 48), focal_length=(70, 75),
        distortion=(0.1, -0.1))
    tof_cam.place((-80, 0, 0))
    tof_cam.scale_resolution(10)
    rgb_cam = CameraModel(chip_size=(40, 30), focal_length=(50, 55),
        distortion=(-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    rgb_cam.place((80, 0, 0))
    rgb_cam.scale_resolution(30)
    cams = [ tof_cam, rgb_cam ]

    # Place light: global lighting
    if True:
        shaders = [ ShaderPointLight((0, 0, 0)) ]
    else:
        shaders = None

    # Visualize scene
    if False:
        objects = []
        for s in scenes:
            objects.append(s)
        for c in cams:
            objects.append(c.get_cs(size=50.0))
            objects.append(c.get_frustum(size=300.0))
        o3d.visualization.draw_geometries(objects)

    # Snap images and save
    for sidx, (scene, pose) in enumerate(zip(scenes, poses)):
        for cidx, cam in enumerate(cams):
            basename = os.path.join(data_dir, f'cam{cidx:02d}_image{sidx:02d}')
            # Snap
            print(f'Snapping image {basename} ...')
            tic = time.monotonic()
            depth_image, color_image, pcl = cam.snap(scene, shaders)
            toc = time.monotonic()
            print(f'    Snapping image took {(toc - tic):.1f}s')
            # Save images
            # Save PCL in camera coodinate system, not in world coordinate system
            pcl.transform(cam.get_pose().inverse().get_homogeneous_matrix())
            save_shot(basename, depth_image, color_image, pcl)
            # Save scene properties
            params = {}
            params['cam'] = {}
            cam.dict_save(params['cam'])
            params['world_to_object'] = {}
            params['world_to_object']['t'] = pose.get_translation().tolist()
            params['world_to_object']['q'] = pose.get_rotation_quaternion().tolist()
            with open(basename + '.json', 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)
