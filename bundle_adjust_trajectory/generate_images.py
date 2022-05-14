import copy
import json
import numpy as np
import open3d as o3d
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import save_shot



def generate_cameras(cam_scale=1.0):
    # cameras
    cameras = []
    # cam 0
    cam0 = CameraModel(chip_size=(40, 30), focal_length=40)
    cam0.place((400, 300, 1000))
    cam0.look_at((80, -50, 10))
    cam0.roll(np.deg2rad(50))
    cameras.append(cam0)
    # cam 1
    cam1 = CameraModel(chip_size=(40, 30), focal_length=20)
    cam1.place((-500, -800, -200))
    cam1.look_at((100, 0, 50))
    cam1.roll(np.deg2rad(-120))
    cameras.append(cam1)
    # cam 2
    cam2 = CameraModel(chip_size=(40, 30), focal_length=35)
    cam2.place((800, -1200, 100))
    cam2.look_at((0, 0, 100))
    cam2.roll(np.deg2rad(85))
    cameras.append(cam2)
    # cam 3
    cam2 = CameraModel(chip_size=(40, 30), focal_length=35)
    cam2.place((-1200, 0, 500))
    cam2.look_at((0, 0, 210))
    cam2.roll(np.deg2rad(-102))
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
#        print(np.hstack((times.reshape((times.size,1)), points)))
    return points



def visualize_scene(sphere, trajectory, cameras, verbose=False):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
    objs = [ cs ]
    for point in trajectory:
        if verbose:
            with np.printoptions(precision=1, suppress=True):
                print(f'trajectory point {point}')
        s = copy.deepcopy(sphere)
        s.translate(point)
        objs.append(s)
    for i, cam in enumerate(cameras):
        if verbose:
            print(f'cam{i}: {cam.get_pose()}')
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
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.paint_uniform_color((0.2, 0.3, 0.4))

    # Create trajectory
    s0 = np.array([200, -300, -150]) # mm
    v0 = np.array([-10, 20, 100]) # mm/s
    ttotal = get_trajectory_total_time(v0)
    dt = 0.5 # Time step in seconds
    num_steps = int(np.ceil(ttotal / dt)) + 1
    times = dt * np.arange(num_steps)
    trajectory = generate_trajectory(times, s0, v0)
    print(f's0 = {s0} mm')
    print(f'v0 = {v0} mm/s')
    print(f'|v0| = {np.linalg.norm(v0):.2f} mm/s')
    print(f'dt = {dt:.2f} s')
    print(f'times = [{times[0]:.2f}, {times[1]:.2f}, .., {times[-1]:.2f}] s')
    print(f'number of samples = {times.size}')

    # Visualize
#    visualize_scene(sphere, trajectory, cameras, verbose=True)

    for step in range(num_steps):
        sphere_center = trajectory[step, :]
        s = copy.deepcopy(sphere)
        s.translate(sphere_center)
        for cam_no, cam in enumerate(cameras):
            basename = os.path.join(data_dir, f'cam{cam_no:02d}_image{step:02d}')
            print(f'Snapping image {basename} ...')
            tic = time.monotonic()
            depth_image, color_image, pcl = cam.snap(s)
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
            params['time'] = times[step]
            with open(basename + '.json', 'w') as f:
                json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')

    # Use Imagemagick to combine images of one camera to a movie
    # convert -delay 500 -quality 100 cam00_image??_color.png cam00.mpg
