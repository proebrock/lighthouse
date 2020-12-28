import copy
import json
import matplotlib.pyplot as plt
plt.close('all')
import matplotlib.cm as cm
import numpy as np
import open3d as o3d
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import save_shot



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
    sphere.scale(sphere_radius / 50.0, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    # Generate sphere centers
    sphere_centers = np.empty((num_spheres, 3))
    sphere_min_dist = 90
    i = 0
    while i < num_spheres:
        sphere_centers[i, 0] = np.random.uniform(-600, 600)
        sphere_centers[i, 1] = np.random.uniform(-400, 400)
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
    elif num_spheres <= 40:
        sphere_colors = generate_distinct_colors40()
    else:
        raise Exception('Not enough colors available')
    spheres = o3d.geometry.TriangleMesh()
    for center, color in zip(sphere_centers, sphere_colors):
        s = copy.deepcopy(sphere)
        s.translate(center)
        s.paint_uniform_color(color)
        spheres += s
    return spheres, sphere_centers



def get_trajectory_total_time(v0):
    # Total time for projectile to return to same height
    g = 9.81
    e_z = np.array([0, 0, 1])
    # Total time for projectile to return to same height
    return (2 * np.dot(v0, e_z)) / g



def generate_trajectory_points(times, s0, v0):
    # Constants
    g = 9.81
    e_z = np.array([0, 0, 1])
    # Generate trajectory
    points = np.empty((times.size, 3))
    for i, t in enumerate(times):
        points[i,:] = s0 + v0 * t - 0.5 * g * t**2 * e_z
    return points



def generate_trajectory(cam_scale=1.0):
    # Generation of trajectory points
    s0 = np.array((-0.4, 0, 0))
    v0 = np.array((1, 0, 1))
    v0 = v0 / np.linalg.norm(v0)
    v0 = 2.82 * v0
    ttotal = get_trajectory_total_time(v0)
    dt = 0.04 # Time step in seconds -> Change here to change number of steps
    num_steps = int(np.floor(ttotal / dt)) + 1
    times = dt * np.arange(num_steps)
    points = generate_trajectory_points(times, s0, v0)
    points = points * 1000.0 # convert unit from m to mm
#    with np.printoptions(precision=3, suppress=True):
#        print(np.hstack((times.reshape((times.size,1)), points)))
    # Generate master cam
    cam = CameraModel(chip_size=(40, 30), focal_length=25)
    cam.scale_resolution(cam_scale)
    # Generate cameras
    cameras = []
    roty = np.linspace(-30, 30, points.shape[0])
    for ry, point in zip(roty, points):
        c = copy.deepcopy(cam)
        T = Trafo3d(t=point, rpy=np.deg2rad((180, ry, 0)))
        c.set_camera_pose(T)
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
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    sphere_radius = 50.0
    num_spheres = 20
    spheres, sphere_centers = generate_spheres(sphere_radius, num_spheres)
    print(f'Scene: {spheres}')

    cam_scale = 40.0
    times, cameras = generate_trajectory(cam_scale)
    print(f'Camera: {cameras[0]}')
    print(f'Number of steps: {times.size}')

#    visualize_scene(spheres, cameras)

    for step, cam in enumerate(cameras):
        # Snap scene
        basename = os.path.join(data_dir, f'cam00_image{step:02d}')
        print(f'Snapping image {basename} ...')
        tic = time.process_time()
        depth_image, color_image, pcl = cam.snap(spheres)
        toc = time.process_time()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
        save_shot(basename, depth_image, color_image, pcl, nan_color=(0, 0, 0))
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