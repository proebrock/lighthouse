import copy
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_3float_to_rgb, image_save
from camsimlib.camera_model import CameraModel



def generate_distinct_colors20():
    steps = np.linspace(0.0, 1.0, 20)
    return cm.tab20(steps)[:,0:3]

def generate_distinct_colors40():
    steps = np.linspace(0.0, 1.0, 20)
    return np.vstack((cm.tab20b(steps), cm.tab20c(steps)))[:,0:3]



def generate_spheres(sphere_radius=50.0, num_spheres=20):
    # Generate master sphere
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    # Generate sphere centers
    sphere_centers = np.empty((num_spheres, 3))
    sphere_min_dist = 90
    i = 0
    while i < num_spheres:
        sphere_centers[i, 0] = np.random.uniform(-500, 500)
        sphere_centers[i, 1] = np.random.uniform(-500, 500)
        sphere_centers[i, 2] = np.random.uniform(-700, -900)
        if i >= 1:
            # Determine distance of new sphere to all other spheres
            dist = np.linalg.norm(sphere_centers[0:i, :]-sphere_centers[i, :], axis=1)
            # If too close to any other sphere, re-generate!
            too_close = np.any(dist < (sphere_min_dist + 2 * sphere_radius))
            if too_close:
                continue
        i += 1
    # Generate spheres
    if num_spheres <= 20:
        sphere_colors = generate_distinct_colors20()
#    elif num_spheres <= 40:
#        sphere_colors = generate_distinct_colors40()
    else:
        raise Exception('Not enough colors available')
    spheres = o3d.geometry.TriangleMesh()
    for center, color in zip(sphere_centers, sphere_colors):
        s = copy.deepcopy(sphere)
        s.translate(center)
        s.paint_uniform_color(color)
        spheres += s
    return spheres, sphere_centers



def generate_trajectory(cam_scale=1.0):
    num_phases = 4
    num_points_per_phase = 6
    num_points = num_phases * num_points_per_phase
    transl = np.zeros((num_points, 3))
    rotrpy = np.zeros((num_points, 3))
    times = np.linspace(0, 10, num_points)
    masks = np.zeros((num_phases, num_points), dtype=bool)
    for i in range(num_phases):
        masks[i,i*num_points_per_phase:(i+1)*num_points_per_phase] = True
    # Phase 0
    transl[masks[0,:],2] = np.linspace(200, 700, num_points_per_phase)
    rotrpy[masks[0,:],0] = 180
    rotrpy[masks[0,:],2] = np.linspace(0, 90, num_points_per_phase)
    # Phase 1
    transl[masks[1,:],0] = np.linspace(0, 700, num_points_per_phase)
    transl[masks[1,:],1] = np.linspace(0, 700, num_points_per_phase)
    transl[masks[1,:],2] = 700
    rotrpy[masks[1,:],0] = np.linspace(180, 210, num_points_per_phase)
    rotrpy[masks[1,:],1] = np.linspace(0, 30, num_points_per_phase)
    rotrpy[masks[1,:],2] = 90
    # Phase 2
    transl[masks[2,:],0] = np.linspace(700, 0, num_points_per_phase)
    transl[masks[2,:],1] = 700
    transl[masks[2,:],2] = np.linspace(700, 200, num_points_per_phase)
    rotrpy[masks[2,:],0] = np.linspace(210, 180, num_points_per_phase)
    rotrpy[masks[2,:],1] = 30
    rotrpy[masks[2,:],2] = 90
    # Phase 3
    transl[masks[3,:],0] = 0
    transl[masks[3,:],1] = np.linspace(700, 0, num_points_per_phase)
    transl[masks[3,:],2] = np.linspace(200, -50, num_points_per_phase)
    rotrpy[masks[3,:],0] = 180
    rotrpy[masks[3,:],1] = np.linspace(30, 0, num_points_per_phase)
    rotrpy[masks[3,:],2] = np.linspace(90, 180, num_points_per_phase)
    # Generate master cam
    cam = CameraModel(chip_size=(40, 30), focal_length=25)
    cam.scale_resolution(cam_scale)
    # Generate cameras
    cameras = []
    for t, rpy in zip(transl, rotrpy):
        c = copy.deepcopy(cam)
        T = Trafo3d(t=t, rpy=np.deg2rad(rpy))
        c.set_pose(T)
        cameras.append(c)
    return times, cameras



def visualize_scene(spheres, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs, spheres ]
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

    sphere_radius = 50.0
    num_spheres = 20
    spheres, sphere_centers = generate_spheres(sphere_radius, num_spheres)
    print(f'Scene: {spheres}')

    cam_scale = 35.0
    times, cameras = generate_trajectory(cam_scale)
    print(f'Camera: {cameras[0]}')
    print(f'Number of steps: {len(cameras)}')
    for cam in cameras:
        pose = cam.get_pose()
        with np.printoptions(precision=2, suppress=True):
            print(f'{pose.get_translation()}, {np.rad2deg(pose.get_rotation_rpy())}')

#    visualize_scene(spheres, cameras)

    for step, cam in enumerate(cameras):
        # Snap scene
        basename = os.path.join(data_dir, f'cam00_image{step:02d}')
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        _, color_image, _ = cam.snap(spheres)
        toc = time.monotonic()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        image = image_3float_to_rgb(color_image)
        image_save(basename + '.png', image)
        # Save all image parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        params['sphere'] = {}
        params['sphere']['center'] = sphere_centers.tolist()
        params['sphere']['radius'] = sphere_radius
        params['time'] = times[step]
        with open(basename + '.json', 'w') as f:
            json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')
