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
    cam1 = CameraModel(chip_size=(40, 30), focal_length=20)
    cam1.place_camera((-500, -800, -200))
    cam1.look_at((100, 0, 50))
    cam1.roll_camera(np.deg2rad(-120))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(40, 30), focal_length=35)
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



def get_trajectory_total_time(v0):
    # Total time for projectile to return to same height
    g = 9.81
    e_z = np.array([0, 0, 1])
    # Total time for projectile to return to same height
    return (2 * np.dot(v0, e_z)) / g



def generate_trajectory(times, s0, v0):
    # Constants
    g = 9.81
    e_z = np.array([0, 0, 1])
    # Generate trajectory
    points = np.empty((times.size, 3))
    for i, t in enumerate(times):
        points[i,:] = s0 + v0 * t - 0.5 * g * t**2 * e_z
#    with np.printoptions(precision=3, suppress=True):
#        print(np.hstack((times.reshape((n,1)), points)))
    return points



def visualize_scene(sphere, trajectory, cameras, verbose=False):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs ]
    for point in trajectory:
        if verbose:
            with np.printoptions(precision=1, suppress=True):
                print(f'trajectory point {point}')
        s = copy.copy(sphere)
        s.translate(point)
        objs.append(s)
    for i, cam in enumerate(cameras):
        if verbose:
            print(f'cam{i}: {cam.get_camera_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
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
    sphere_radius = 50.0
    sphere.scale(sphere_radius / 50.0, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    print('sphere bbox min', np.min(np.asarray(sphere.vertices), axis=0))
    print('sphere bbox max', np.max(np.asarray(sphere.vertices), axis=0))
    sphere.paint_uniform_color((0.2, 0.3, 0.4))

    # Create trajectory
    s0 = np.array([0.2, -0.3, -0.15])
    v0 = np.array([-0.1, 0.2, 1])
    v0 = v0 / np.linalg.norm(v0)
    v0 = 3.5 * v0
    ttotal = get_trajectory_total_time(v0)
    dt = 0.025 # Time step in seconds
    num_steps = int(np.ceil(ttotal / dt)) + 1
    times = dt * np.arange(num_steps)
    points = generate_trajectory(times, s0, v0)
    trajectory = points * 1000.0 # convert unit from m to mm

    # Visualize
#    visualize_scene(sphere, trajectory, cameras, verbose=True)

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
            params['time'] = times[step]
            with open(basename + '.json', 'w') as f:
               json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')

    # Use Imagemagick to combine images of one camera to a movie
    # convert -delay 2.5 -quality 100 cam00_image??_color.png cam00.mpg
