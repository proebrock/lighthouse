import cv2
import json
import numpy as np
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.charuco_board import CharucoBoard
from camsimlib.scene_visualizer import SceneVisualizer



data_dir = 'b'



def generate_calibration_views(mesh, n_views):
    # We assume the calibration plate is in X/Y plane with Z=0
    mesh_min = np.min(mesh.vertices, axis=0)
    mesh_max = np.max(mesh.vertices, axis=0)
    # Pick point to look at on the surface of the mesh
    look_at_pos = np.zeros((n_views, 3))
    look_at_pos[:,0] = np.random.uniform(mesh_min[0], mesh_max[0], n_views)
    look_at_pos[:,1] = np.random.uniform(mesh_min[1], mesh_max[1], n_views)
    # Pick a camera position in X/Y
    cam_scale = 3.0
    camera_pos = np.zeros((n_views, 3))
    camera_pos[:,0] = np.random.uniform(cam_scale*mesh_min[0], cam_scale*mesh_max[0], n_views)
    camera_pos[:,1] = np.random.uniform(cam_scale*mesh_min[1], cam_scale*mesh_max[1], n_views)
    # Camera Z position is determined by desired view angle
    phi_min = np.deg2rad(40)
    phi_max = np.deg2rad(60)
    phi = np.random.uniform(phi_min, phi_max, n_views)
    camera_pos[:,2] = np.linalg.norm(look_at_pos - camera_pos, axis=1) * np.tan(phi)
    # Unit vector in Z is direction from camera to view point
    ez = look_at_pos - camera_pos
    ez /= np.linalg.norm(ez, axis=1).reshape(n_views,1)
    # Unit vector in Y is perpendicular to ez
    ey = np.random.uniform(-1.0, 1.0, (n_views, 3))
    ey[:,2] = (- ey[:,0] * ez[:,0] - ey[:,1] * ez[:,1]) / ez[:,2]
    ey /= np.linalg.norm(ey, axis=1).reshape(n_views,1)
    # Unit vector in X is perpendicular to ey and ez
    ex = np.cross(ey, ez, axis=1)
    # Assemble transformations
    trafos = []
    for i in range(n_views):
        R = np.vstack((ex[i,:], ey[i,:], ez[i,:])).T
        T = Trafo3d(t=camera_pos[i,:], mat=R)
        trafos.append(T)
    return trafos



def move_cs_along_z(T, alpha):
    R = T.get_rotation_matrix()
    return Trafo3d(t=T.get_translation()-alpha*R[:, 2], mat=R)


def objfun(x, cam, mesh, trafo):
    pass


def generate_calibration_views2(cam, mesh, n_views):
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

    trafos = []
    for i in range(n_views):
        T = Trafo3d(t=look_at_pos[i,:], rpy=rpy[i,:])
        trafos.append(T)
    return trafos



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



np.random.seed(42)
cam = CameraModel(pix_size=(120, 90), f=(150, 140), c=(75, 66))
mesh = CharucoBoard((6,5), 30.0)
trafos = generate_calibration_views(cam, mesh, 10)



#np.random.seed(42)
#board = CharucoBoard((6,5), 30.0)
##board.show(True, False, False)
#trafos = generate_calibration_views(board, 10)
#a = 1 # Use this scale factor to control image size and computation time
#cams = [ CameraModel(pix_size=(160*a, 120*a), f=(200*a,190*a),
#                     c=(80*a,63*a), trafo=T) for T in trafos ]
#show_calibration_views(board, cams)

#for i, cam in enumerate(cams):
#    print(f'Snapping image {i+1}/{len(trafos)} ...')
#    tic = time.process_time()
#    dImg, cImg, P = cam.snap(board)
#    toc = time.process_time()
#    print(f'    Snapping image took {(toc - tic):.1f}s')
#    # Save image
#    basename = os.path.join(data_dir, f'image{i:02d}')
#    save_image(basename + '.png', cImg)
#    # Save all image parameters
#    params = {}
#    params['cam'] = {}
#    cam.dict_save(params['cam'])
#    params['board'] = {}
#    board.dict_save(params['board'])
#    with open(basename + '.json', 'w') as f:
#       json.dump(params, f, indent=4, sort_keys=True)
#
#print('Done.')

