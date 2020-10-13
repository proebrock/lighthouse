import copy
import cv2
import json
import numpy as np
import os
import sys
import time
from scipy.optimize import minimize_scalar

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.charuco_board import CharucoBoard
from camsimlib.mesh_plane import MeshPlane
from camsimlib.scene_visualizer import SceneVisualizer



def move_pose_along_z(T, alpha):
    R = T.get_rotation_matrix()
    return Trafo3d(t=T.get_translation()-alpha*R[:, 2], mat=R)



def objfun(x, cam, mesh, trafo):
    # Generate a camera with a chip twice as big and at the given pose
    hicam = copy.deepcopy(cam)
    hicam.set_chip_size(3 * hicam.get_chip_size())
    hicam.set_principal_point(3 * hicam.get_principal_point())
    hicam.set_camera_pose(move_pose_along_z(trafo, x))
    # Snap image
    depth_image, color_image, P = hicam.snap(mesh)
    # Analyze image
    h, w = cam.get_chip_size()
    inner_image = depth_image[h:2*h,w:2*w]
    good_pixel_count = np.sum(~np.isnan(inner_image))
    bad_pixel_count = np.sum(~np.isnan(depth_image)) - good_pixel_count
    return 2 * (bad_pixel_count//8) - good_pixel_count



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

    poses = []
    for i in range(n_views):
        print(f'Generating view {i+1}/{n_views} ...')
        T = Trafo3d(t=look_at_pos[i,:], rpy=rpy[i,:])
        options = { 'xatol': 1e-01, 'maxiter': 50 }
        res = minimize_scalar(objfun, args=(cam, mesh, T),
                              bounds=(1, 5000), method='bounded',
                              options=options)
        print(res)
        if not res.success:
            raise Exception('Optimization unsuccessful')
        T = move_pose_along_z(T, res.x)
        poses.append(T)
    return poses



def show_calibration_views(board, cams):
    vis = SceneVisualizer()
    vis.add_mesh(board)
    for cam in cams:
        vis.add_cam_cs(cam, size=100.0)
        #vis.add_cam_frustum(cam, size=600.0)
    vis.show()



def save_image(filename, img):
    # Find NaN values
    nanidx = np.where(np.isnan(img))
    # Convert image to integer
    img = (255.0 * img).astype(np.uint8)
    # Convert RGB to gray image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Maximize dynamic range
    img = ((img - np.min(img)) * 255.0) / (np.max(img) - np.min(img))
    # Set NaN to distinct color
    img[nanidx[0],nanidx[1]] = 127
    # Write image
    cv2.imwrite(filename, img)



np.random.seed(42) # Random but reproducible
data_dir = 'b'
if not os.path.exists(data_dir):
    raise Exception('Target directory does not exist.')
# Generate camera; resolution must be quite low
cam = CameraModel(chip_size=(40, 30), focal_length=(50, 50))
# Generate calibration board
squares = (6, 5)
square_length = 30.0
board = CharucoBoard(squares, square_length)
plane = MeshPlane(square_length * np.array(squares), color=(1,0,1))
poses = generate_calibration_camera_poses(cam, plane, 3)
cams = []
for pose in poses:
    c = copy.deepcopy(cam)
    c.scale_resolution(5) # Scale up camera resolution
    c.set_camera_pose(pose) # Assign previously generated pose
    cams.append(c)
#show_calibration_views(board, cams)

for i, cam in enumerate(cams):
    print(f'Snapping image {i+1}/{len(cams)} ...')
    tic = time.process_time()
    dImg, cImg, P = cam.snap(board)
    toc = time.process_time()
    print(f'    Snapping image took {(toc - tic):.1f}s')
    # Save image
    basename = os.path.join(data_dir, f'image{i:02d}')
    save_image(basename + '.png', cImg)
    # Save all image parameters
    params = {}
    params['cam'] = {}
    cam.dict_save(params['cam'])
    params['board'] = {}
    board.dict_save(params['board'])
    with open(basename + '.json', 'w') as f:
       json.dump(params, f, indent=4, sort_keys=True)
print('Done.')

