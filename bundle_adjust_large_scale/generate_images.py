import copy
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_3float_to_rgb, image_save
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from color_matcher import generate_colors



def visualize_scene(cams, balls):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ cs ]
    for cam in cams:
        objects.append(cam.get_cs(size=50))
        objects.append(cam.get_frustum(size=200))
    for ball in balls:
        objects.append(ball)
    o3d.visualization.draw_geometries(objects)



if __name__ == '__main__':
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Generate camera model
    cam = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam.scale_resolution(40)

    if False:
        # Random camera placements

        # Generate 3D points
        num_points = 20
        P = np.random.uniform(-200, 200, (num_points, 3))
        num_views = 40
        world_to_cam_1 = Trafo3d(t=(0, 0, -1000))
        poses = []
        for i in range(num_views):
            # Small movements in point coordinate system
            t = np.random.uniform(-50, 50, 3)
            rpy = np.random.uniform(-180, 180, 3)
            # Transformed into camera movement
            points_trafo = Trafo3d(t=t, rpy=np.deg2rad(rpy))
            world_to_cam_n = points_trafo * world_to_cam_1
            poses.append(world_to_cam_n)
        # Create individual camera per view
        cams = []
        for T in poses:
            c = copy.deepcopy(cam)
            c.set_pose(T)
            cams.append(c)
    else:
        # Spiral-shaped continuous camera movement

        # Generate 3D points
        num_points = 40
        P = np.random.uniform(-200, 200, (num_points, 3))
        num_views = 40
        height_max = 1000
        height_min = 200
        radius = 500
        angles = np.linspace(0, 2*np.pi, num_views + 1)[0:-1]
        cams = []
        poses = []
        for i, angle in enumerate(angles):
            c = copy.deepcopy(cam)
            c.place((
                radius * np.cos(angle),
                radius * np.sin(angle),
                height_min + ((height_max - height_min) * i) / num_views
                ))
            c.look_at((0, 0, 0))
            cams.append(c)
            poses.append(c.get_pose())

    spheres = []
    colors = generate_colors()
    #plot_colorbar(colors)
    for i in range(P.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
        sphere.paint_uniform_color(colors[i, :])
        sphere.translate(P[i, :])
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        spheres.append(sphere)
    all_spheres_mesh = o3d.geometry.TriangleMesh()
    for sphere in spheres:
        all_spheres_mesh += sphere

    #visualize_scene(cams, balls)

    for i, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'cam{i:02d}')
        # Snap scene
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        _, image, _ = cam.snap(all_spheres_mesh)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        image = image_3float_to_rgb(image)
        image_save(basename + '.png', image)
        # Save parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        # Save sphere colors
        params['sphere_colors'] = colors.tolist()
        # Save 3D sphere locations
        params['sphere_centers'] = P.tolist()
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)

    print('Done.')
