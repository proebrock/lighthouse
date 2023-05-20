import copy
import numpy as np
import open3d as o3d
import json
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load, image_3float_to_rgb, image_save
from common.mesh_utils import pcl_save
from camsimlib.camera_model import CameraModel
from camsimlib.screen import Screen
from trafolib.trafo3d import Trafo3d
from camsimlib.shader_point_light import ShaderPointLight



def generate_scene():
    image = image_load('../data/lena.jpg')
    dimension = 0.2 * np.array((image.shape[1], image.shape[0]))
    screen0 = Screen(dimension, image, pose=Trafo3d(t=(0, 0-102.4, 500)))
    screen1 = Screen(dimension, image, pose=Trafo3d(t=(100, 150-102.4, 800)))
    screen2 = Screen(dimension, image, pose=Trafo3d(t=(-150, -150-102.4, 1000)))
    screen3 = Screen(dimension, image, pose=Trafo3d(t=(-150, 200-102.4, 1200)))
    screens = [ screen0, screen1, screen2, screen3 ]
    mesh = o3d.geometry.TriangleMesh()
    for screen in screens:
        mesh += screen.get_mesh()
    return mesh, screens



def visualize_scene(screens, cams):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100)
    objects = [ cs ]
    for s in screens:
        objects.append(s.get_cs(size=50.0))
        objects.append(s.get_mesh())
    for c in cams:
        objects.append(c.get_cs(size=50.0))
        objects.append(c.get_frustum(size=300.0))
    o3d.visualization.draw_geometries(objects)



def snap_and_save(cams, mesh, title, shaders):
    # Snap images and save
    for cidx, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'{title}_cam{cidx:02d}')
        # Snap
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        _, image, pcl = cam.snap(mesh, shaders)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        image = image_3float_to_rgb(image)
        image_save(basename + '.png', image)
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(cam.get_pose().inverse().get_homogeneous_matrix())
        pcl_save(basename + '.ply', pcl)
        # Save parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Generate scene
    mesh, screens = generate_scene()

    # Generate cameras
    cam_left = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_left.place((-40, 0, 0))
    cam_left.scale_resolution(30)
    cam_right = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_right.place((40, 0, 0))
    cam_right.scale_resolution(30)
    cams = [ cam_left, cam_right ]

    #visualize_scene(screens, cams)

    # Place light: global lighting
    if True:
        shaders = [ ShaderPointLight((0, 0, 0)) ]
    else:
        shaders = None

    # Perfect setting
    snap_and_save(cams, mesh, 'ideal', shaders)

    # Realistic setting: Distorted
    cams[0].set_distortion((0.2, -0.2))
    cams[1].set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    snap_and_save(cams, mesh, 'distorted', shaders)
    cams[0].set_distortion((0.0, 0.0))
    cams[1].set_distortion((0.0, 0.0))

    # Realistic setting: Right camera slightly displaced (translated or rotated)
    T_orig = cams[1].get_pose()
    T = T_orig * Trafo3d(t=(5, 0, 0), rpy=np.deg2rad((0, 0, 0)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_tx', shaders)
    T = T_orig * Trafo3d(t=(0, 5, 0), rpy=np.deg2rad((0, 0, 0)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_ty', shaders)
    T = T_orig * Trafo3d(t=(0, 0, 5), rpy=np.deg2rad((0, 0, 0)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_tz', shaders)
    T = T_orig * Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((2, 0, 0)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_rx', shaders)
    T = T_orig * Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 2, 0)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_ry', shaders)
    T = T_orig * Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 0, 2)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced_rz', shaders)

    T = T_orig * Trafo3d(t=(7, 3, -14), rpy=np.deg2rad((-1.5, 3, 2)))
    cams[1].set_pose(T)
    snap_and_save(cams, mesh, 'displaced', shaders)

    # Realistic setting: Distorted and displaced
    cams[0].set_distortion((0.2, -0.2))
    cams[1].set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    snap_and_save(cams, mesh, 'distorted_displaced', shaders)

    print('Done.')
