import cv2
import cv2.aruco as aruco
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
plt.close('all')
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d



def determine_correspondences(obj_ids, obj_points, img_ids, img_points):
    assert obj_ids.size == obj_points.shape[0]
    assert obj_points.shape[1] == 4 # Four corners per marker
    assert obj_points.shape[2] == 3 # 3D points
    assert img_ids.size == img_points.shape[0]
    assert img_points.shape[1] == 4 # Four corners per marker
    assert img_points.shape[2] == 2 # 2D points
    unique_ids = np.unique(np.concatenate((img_ids, obj_ids)))
    P = np.zeros((unique_ids.size, 4, 3))
    p = np.zeros((unique_ids.size, 4, 2))
    for i in range(unique_ids.size):
        obj_idx = np.where(obj_ids == unique_ids[i])[0]
        P[i, :, :] = obj_points[obj_idx, :, :]
        img_idx = np.where(img_ids == unique_ids[i])[0]
        p[i, :, :] = img_points[img_idx, :, :]
    P = P.reshape((4 * unique_ids.size, 3))
    p = p.reshape((4 * unique_ids.size, 2))
    return P, p



def solve_pnp_objfun(x, P, p ,cam):
    T = Trafo3d(t=x[:3], rodr=x[3:])
    cam.set_camera_pose(T)
    p_proj = cam.scene_to_chip(P)
    p_proj = p_proj[:,0:2] # Omit the distance information
    return (p - p_proj).ravel()




def solve_pnp(P, p, cam):
    assert P.shape[0] == p.shape[0]
    assert P.shape[1] == 3
    assert p.shape[1] == 2
    x0 = np.zeros(6)
    result = least_squares(solve_pnp_objfun, x0, args=(P, p, cam))
    if not result.success:
        raise Exception('solve_pnp failed: ' + str(result))
    cam_to_obj_estim = Trafo3d(t=result.x[:3], rodr=result.x[3:]).inverse()
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(121)
        residuals = solve_pnp_objfun(result.x, P, p, cam)
        ax.plot(residuals)
        ax.set_ylabel('Residuals (mm)')
        ax.grid()
        ax = fig.add_subplot(122)
        dist = np.linalg.norm(residuals.reshape((-1, 2)), axis=1)
        ax.plot(dist)
        ax.set_xlabel('Point index')
        ax.set_ylabel('Pixel distances (mm)')
        ax.grid()
        plt.show()
    return cam_to_obj_estim



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/leafstring/multi-marker'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    basename = 'cam00_image00'

    # Load camera and other settings
    with open(os.path.join(data_dir, basename + '.json'), 'r') as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    world_to_plane = Trafo3d(t=params['plane_pose']['t'],
                             q=params['plane_pose']['q'])
    world_to_cam = cam.get_camera_pose()
    cam.set_camera_pose(Trafo3d()) # Remove solution from camera object
    cam_to_plane = world_to_cam.inverse() * world_to_plane
    marker_ids = []
    marker_coords = []
    for key, value in params['markers'].items():
        marker_ids.append(int(key))
        marker_coords.append(value['coords'])
    obj_ids = np.asarray(marker_ids, dtype=np.int)
    obj_points = np.asarray(marker_coords)

    # Load image
    img = cv2.imread(os.path.join(data_dir, basename + '_color.png'))

    # Detect markers
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    if True:
        display_img = img.copy()
        aruco.drawDetectedMarkers(display_img, corners, ids)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(display_img)
        plt.show()
    img_ids = np.asarray(ids, dtype=np.int).reshape((-1, ))
    img_points = np.asarray(corners).reshape((-1, 4, 2))

    # Get correspondences of object and image points based in IDs
    P, p = determine_correspondences(obj_ids, obj_points, img_ids, img_points)

    # Solve point-to-point problem
    cam_to_plane_estim = solve_pnp(P, p, cam)
    print(f'cam_T_plane:\n    {cam_to_plane}')
    print(f'cam_T_plane estimated:\n    {cam_to_plane_estim}')
