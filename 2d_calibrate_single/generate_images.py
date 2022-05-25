import copy
import json
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
plt.close('all')
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_generate_plane, \
    mesh_generate_charuco_board, save_shot



def move_pose_along_z(T, alpha):
    R = T.get_rotation_matrix()
    t = T.get_translation() - alpha * R[:, 2]
    return Trafo3d(t=t, mat=R)



def objfun(x, cam, mesh, pose):
    # Generate a camera with a chip 3x3 times as big
    hicam = copy.deepcopy(cam)
    hicam.set_chip_size(3 * hicam.get_chip_size())
    hicam.set_principal_point(3 * hicam.get_principal_point())
    hicam.set_pose(move_pose_along_z(pose, x))
    # Snap image
    depth_image, color_image, pcl = hicam.snap(mesh)
    # Analyze image
    w, h = cam.get_chip_size()
    # The original camera image is in the center of the 3x3
    inner_image = depth_image[h:2*h,w:2*w]
    # We want as many valid image points as possible in inner_image
    good_pixel_count = np.sum(~np.isnan(inner_image))
    # But as little as possible in the outer (3x3)-1 region
    bad_pixel_count = np.sum(~np.isnan(depth_image)) - good_pixel_count
    # Debug output
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(depth_image)
        ax.axvline(x=w, color='r')
        ax.axvline(x=2*w, color='r')
        ax.axhline(y=h, color='r')
        ax.axhline(y=2*h, color='r')
        ax.text(0, 0, f'bad:{bad_pixel_count}', va='top', size=10, color='k')
        ax.text(w, h, f'good:{good_pixel_count}', va='top', size=10, color='k')
        ax.set_title(f'x={x}')
        plt.show()
    #return good_pixel_count, bad_pixel_count
    return bad_pixel_count - good_pixel_count



def analyze_objective_function(cam, mesh, pose):
    x = np.linspace(1, 1000.0, 21)
    y = np.zeros((x.size, 2))
    for i in range(x.size):
        print(f'{i+1}/{x.size} ...')
        y[i,:] = objfun(x[i], cam, mesh, pose)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x, y[:,0], '-og')
    ax.plot(x, y[:,1], '-or')
    ax.plot(x, y[:,1]-y[:,0], '-ob')
    ax.grid()
    plt.show()



def generate_calibration_camera_poses(cam, mesh, n_views):
    # We assume the calibration plate is in X/Y plane with Z=0
    mesh_min = np.min(mesh.vertices, axis=0)
    mesh_max = np.max(mesh.vertices, axis=0)
    # Pick point to look at on the surface of the mesh (assuming a plane here)
    look_at_pos = np.zeros((n_views, 3))
    look_at_pos[:,0] = np.random.uniform(mesh_min[0], mesh_max[0], n_views) # X
    look_at_pos[:,1] = np.random.uniform(mesh_min[1], mesh_max[1], n_views) # Y
    # Rotate coordinate system that its z axis is the camera view direction
    rpy = np.zeros((n_views, 3))
    phi = 50.0
    rpy[:,0] = np.deg2rad(180 + np.random.uniform(-phi, phi, n_views)) # rot X
    rpy[:,1] = np.deg2rad(np.random.uniform(-phi, phi, n_views)) # rot Y
    rpy[:,2] = np.random.uniform(-np.pi, np.pi, n_views) # rot Z

    #i = 13
    #objfun(764, cam, mesh, Trafo3d(t=look_at_pos[i,:], rpy=rpy[i,:]))
    #analyze_objective_function(cam, mesh, Trafo3d(t=look_at_pos[i,:], rpy=rpy[i,:]))

    poses = []
    for i in range(n_views):
        print(f'Generating view {i+1}/{n_views} ...')
        T = Trafo3d(t=look_at_pos[i,:], rpy=rpy[i,:])
        options = { 'xatol': 1.0, 'maxiter': 50 }
        tic = time.monotonic()
        res = minimize_scalar(objfun, args=(cam, mesh, T),
                              bounds=(1, 5000), method='bounded',
                              options=options)
        toc = time.monotonic()
        if not res.success:
            print(res)
            raise Exception('Optimization unsuccessful')
        print(f'    took snapping of {res.nfev} images and {(toc - tic):.1f}s time')
        T = move_pose_along_z(T, res.x)
        poses.append(T)
    return poses



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Generate camera; resolution must be quite low
    cam = CameraModel(chip_size=(40, 30), focal_length=(50, 55),
                      distortion=(-0.8, 0.8))
    # Generate calibration board
    squares = (6, 5)
    square_length = 30.0
    board = mesh_generate_charuco_board(squares, square_length)
    plane = mesh_generate_plane(square_length * np.array(squares), color=(1,0,1))
    poses = generate_calibration_camera_poses(cam, plane, 16)
    cams = []
    for pose in poses:
        c = copy.deepcopy(cam)
        c.scale_resolution(30) # Scale up camera resolution
        c.set_pose(pose) # Assign previously generated pose
        cams.append(c)

    for i, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'image{i:02d}')
        print(f'Snapping image {basename} ...')
        tic = time.monotonic()
        depth_image, color_image, pcl = cam.snap(board)
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
        params['board'] = {}
        params['board']['squares'] = squares
        params['board']['square_length'] = square_length
        with open(basename + '.json', 'w') as f:
           json.dump(params, f, indent=4, sort_keys=True)
    print('Done.')
