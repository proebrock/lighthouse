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
    cam0.place_camera((400, 300, 1000))
    cam0.look_at((80, -50, 10))
    cam0.roll_camera(np.deg2rad(50))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=25)
    cam1.place_camera((-500, -800, -200))
    cam1.look_at((100, 0, 50))
    cam1.roll_camera(np.deg2rad(-120))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(40, 30), focal_length=30)
    cam2.place_camera((800, -1200, 100))
    cam2.look_at((0, 0, 100))
    cam2.roll_camera(np.deg2rad(85))
    cameras.append(cam2)
    # cam 3
    cam2 = CameraModel(chip_size=(40, 30), focal_length=35)
    cam2.place_camera((-1200, 0, 500))
    cam2.look_at((0, 0, 210))
    cam2.roll_camera(np.deg2rad(-102))
    cameras.append(cam2)
    # Scale cameras
    for cam in cameras:
        cam.scale_resolution(cam_scale)
    return cameras



def generate_trajectory(s0, v0, n):
    # Constants
    g = 9.81
    e_z = np.array([0, 0, 1])
    # Total time for projectile to return to same height
    t_total = (2 * np.dot(v0, e_z)) / g
    # Generate time stamps
    times = np.linspace(0.0, t_total, n)
    # Generate trajectory
    points = np.empty((n, 3))
    for i, t in enumerate(times):
        points[i,:] = s0 + v0 * t - 0.5 * g * t**2 * e_z
#    with np.printoptions(precision=3, suppress=True):
#        print(np.hstack((times.reshape((n,1)), points)))
    return times, points



def visualize_scene(sphere, trajectory, cameras):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs ]
    for point in trajectory:
        with np.printoptions(precision=1, suppress=True):
            print(f'trajectory point {point}')
        s = copy.copy(sphere)
        s.translate(point)
        objs.append(s)
    for i, cam in enumerate(cameras):
        print(f'cam{i}: {cam.get_camera_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



np.random.seed(42) # Random but reproducible
data_dir = 'a'
if not os.path.exists(data_dir):
    raise Exception('Target directory does not exist.')

# Setup cameras
cameras = generate_cameras(cam_scale=30.0)

# Create sphere
sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
sphere.compute_triangle_normals()
sphere.compute_vertex_normals()
sphere.scale(0.5, center=sphere.get_center())
sphere.translate(-sphere.get_center())
print('sphere bbox min', np.min(np.asarray(sphere.vertices), axis=0))
print('sphere bbox max', np.max(np.asarray(sphere.vertices), axis=0))
sphere_radius = 25.0
sphere.paint_uniform_color((0.2, 0.3, 0.4))

# Create trajectory
num_steps = 21
s0 = np.array([0.2, -0.3, -0.15])
v0 = np.array([-0.1, 0.2, 1])
v0 = v0 / np.linalg.norm(v0)
v0 = 3.5 * v0
times, points = generate_trajectory(s0, v0, num_steps)
trajectory = points * 1000.0 # convert unit from m to mm

# Visualize
visualize_scene(sphere, trajectory, cameras)

for step in range(num_steps):
    sphere_center = trajectory[step, :]
    s = copy.copy(sphere)
    s.translate(sphere_center)
    for cam_no, cam in enumerate(cameras):
        basename = os.path.join(data_dir, f'cam{cam_no:02d}_image{step:02d}')
        print(f'Snapping image {basename} ...')
        tic = time.process_time()
        depth_image, color_image, pcl = cam.snap(s)
        toc = time.process_time()
        print(f'    Snapping image took {(toc - tic):.1f}s')
        # Save generated snap
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
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
