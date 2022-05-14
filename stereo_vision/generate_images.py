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
from camsimlib.o3d_utils import mesh_transform, mesh_generate_image_file, save_shot
from camsimlib.shader_point_light import ShaderPointLight



def generate_images(position):
    mesh = mesh_generate_image_file('../data/lena.jpg', pixel_size=1.0, scale=0.2)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, Trafo3d(rpy=np.deg2rad([0, 180, 180])))
    mesh_pose = Trafo3d(t=position)
    mesh_transform(mesh, mesh_pose)
    return mesh



def visualize_scene(mesh, cams):
    objects = []
    objects.append(mesh)
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
        depth_image, color_image, pcl = cam.snap(mesh, shaders)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save images
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(cam.get_pose().inverse().get_homogeneous_matrix())
        save_shot(basename, depth_image, color_image, pcl, nan_color=(0, 0, 0))
        # Save scene properties
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Generate scenes
    mesh = generate_images((0, 0, 500))
    mesh += generate_images((100, 150, 800))
    mesh += generate_images((-150, -150, 1000))
    mesh += generate_images((-150, 200, 1200))

    # Generate cameras
    cam_left = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_left.place((-40, 0, 0))
    cam_left.scale_resolution(30)
    cam_right = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_right.place((40, 0, 0))
    cam_right.scale_resolution(30)
    cams = [ cam_left, cam_right ]

    # Place light: global lighting
    if True:
        shaders = [ ShaderPointLight((0, 0, 0)) ]
    else:
        shaders = None

    # Perfect setting
    #visualize_scene(mesh, cams)
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
