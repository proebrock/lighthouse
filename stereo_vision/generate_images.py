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
from camsimlib.o3d_utils import mesh_transform, mesh_generate_image, save_shot



def generate_plane(position, color=(0, 255, 0)):
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :] = color
    mesh = mesh_generate_image(img, pixel_size=10.0)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, Trafo3d(rpy=np.deg2rad([0, 180, 0])))
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



def snap_and_save(cams, mesh, title):
    # Snap images and save
    for cidx, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'{title}_cam{cidx:02d}')
        # Snap
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        depth_image, color_image, pcl = cam.snap(mesh)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save images
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
        save_shot(basename, depth_image, color_image, pcl, nan_color=(0, 0, 0))
        # Save scene properties
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'c'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Generate scenes
    mesh = generate_plane((0, 0, 500), (255, 0, 0)) # Red
    mesh += generate_plane((100, 150, 800), (0, 255, 0)) # Green
    mesh += generate_plane((-150, -150, 1000), (0, 0, 255)) # Blue
    mesh += generate_plane((-150, 200, 1200), (0, 255, 255)) # Cyan

    # Generate cameras
    cam_left = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_left.place_camera((-50, 0, 0))
    cam_left.scale_resolution(30)
    cam_right = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
    cam_right.place_camera((50, 0, 0))
    cam_right.scale_resolution(30)
    cams = [ cam_left, cam_right ]

    # Place light: global lighting
    if True:
        lighting_mode = 'point'
        light_vector = (0, 0, 0)
        cam_left.set_lighting_mode(lighting_mode)
        cam_left.set_light_vector(light_vector)
        cam_right.set_lighting_mode(lighting_mode)
        cam_right.set_light_vector(light_vector)

    # Perfect setting
    #visualize_scene(mesh, cams)
    snap_and_save(cams, mesh, 'ideal')

    # Realistic setting: Distortion
    cams[0].set_distortion((0.1, -0.1))
    cams[1].set_distortion((-0.1, 0.1))
    # Realistic setting: Right camera slightly moved
    T = cams[1].get_camera_pose()
    T = T * Trafo3d(t=(7, 3, -14), rpy=np.deg2rad((-1.5, 3, 2)))
    cams[1].set_camera_pose(T)
    #visualize_scene(mesh, cams)
    snap_and_save(cams, mesh, 'realistic')

