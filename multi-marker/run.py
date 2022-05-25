import cv2
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
    common_ids = np.intersect1d(img_ids, obj_ids)
    P = np.zeros((common_ids.size, 4, 3))
    p = np.zeros((common_ids.size, 4, 2))
    for i in range(common_ids.size):
        obj_idx = np.where(obj_ids == common_ids[i])[0]
        P[i, :, :] = obj_points[obj_idx, :, :]
        img_idx = np.where(img_ids == common_ids[i])[0]
        p[i, :, :] = img_points[img_idx, :, :]
    P = P.reshape((4 * common_ids.size, 3))
    p = p.reshape((4 * common_ids.size, 2))
    return P, p



def solve_pnp_objfun(x, P, p ,cam):
    T = Trafo3d(t=x[:3], rodr=x[3:])
    cam.set_pose(T)
    p_proj = cam.scene_to_chip(P)
    p_proj = p_proj[:,0:2] # Omit the distance information
    return (p - p_proj).ravel()



def solve_pnp(P, p, cam, x0_trafo=Trafo3d(), verbose=False):
    assert P.shape[0] == p.shape[0]
    assert P.shape[1] == 3
    assert p.shape[1] == 2
    x0 = np.concatenate((x0_trafo.get_translation(), x0_trafo.get_rotation_rodrigues()))
    result = least_squares(solve_pnp_objfun, x0, args=(P, p, cam))
    if not result.success:
        raise Exception('solve_pnp failed: ' + str(result))
    cam_to_obj_estim = Trafo3d(t=result.x[:3], rodr=result.x[3:]).inverse()
    if verbose:
        residuals = solve_pnp_objfun(result.x, P, p, cam)
        dist = np.linalg.norm(residuals.reshape((-1, 2)), axis=1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(dist)
        ax.set_xlabel('Point index')
        ax.set_ylabel('Reprojection errors (pixels)')
        ax.grid()
        plt.show()
    return cam_to_obj_estim



def detect_markers(img):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    # Corner refinement method: Determines speed and accuracy of detection
#    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR # use contour-Points
#    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX #  do subpixel refinement
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # use the AprilTag2 approach
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict,
        parameters=aruco_params)
    img_ids = np.asarray(ids, dtype=int).reshape((-1, ))
    img_points = np.asarray(corners).reshape((-1, 4, 2))
    return img_ids, img_points



def detect_markers2(img, cam, square_length, object_to_markers):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG # use the AprilTag2 approach
    # Detecting markers with given camera matrix and distortion coeffs is more successful
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, aruco_dict,
        parameters=aruco_params, cameraMatrix=cam.get_camera_matrix(),
        distCoeff=cam.get_distortion())
    img_ids = np.asarray(ids, dtype=int).reshape((-1, ))
    img_points = np.asarray(corners).reshape((-1, 4, 2))
    # We estimate the pose of each marker
    rvecs, tvecs, _obj_points = cv2.aruco.estimatePoseSingleMarkers(corners, square_length,
        cameraMatrix=cam.get_camera_matrix(), distCoeffs=cam.get_distortion())
    # Calculate cam_to_obj for each marker; we use this as a good initial value x0_trafo
    # for the numerical optimization done in solve_pnp() later
    cam_to_obj_estim = []
    for i in range(len(rvecs)):
        cam_to_marker = Trafo3d(t=tvecs[i], rodr=rvecs[i])
        if not img_ids[i] in object_to_markers:
            raise Exception(f'Unable to find {img_ids[i]} in object_to_markers dict.')
        cam_to_obj_estim.append(cam_to_marker * object_to_markers[img_ids[i]].inverse())
    average, errors = Trafo3d.average_and_errors(cam_to_obj_estim)
    return img_ids, img_points, average



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'multi-marker')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    basename = 'cam00_image00'

    # Load camera and other settings
    with open(os.path.join(data_dir, basename + '.json'), 'r') as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    world_to_object = Trafo3d(t=params['world_to_object']['t'],
                              q=params['world_to_object']['q'])
    world_to_cam = cam.get_pose()
    cam.set_pose(Trafo3d()) # Remove solution from camera object
    cam_to_object = world_to_cam.inverse() * world_to_object # The expected solution
    marker_ids = []
    obj_points = []
    square_length = None
    object_to_markers = {}
    for key, value in params['markers'].items():
        marker_id = int(key)
        marker_ids.append(marker_id)
        object_to_marker = Trafo3d(t=value['object_to_marker']['t'],
                                   q=value['object_to_marker']['q'])
        object_to_markers[marker_id] = object_to_marker
        marker_points = np.asarray(value['points'])
        obj_points.append(object_to_marker * marker_points)
        square_length_ = float(np.asarray(value['square_length']))
        if square_length is None:
            square_length = square_length_
        else:
            assert np.isclose(square_length, square_length_)
    obj_ids = np.asarray(marker_ids, dtype=int)
    obj_points = np.asarray(obj_points)

    # Show object points
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for oid, opoints in zip(obj_ids, obj_points):
            ax.plot(opoints[:,0], opoints[:,1], 'xr')
            m = np.mean(opoints, axis=0)
            ax.text(m[0], m[1], f'Marker {oid}',
                horizontalalignment='center', verticalalignment='center')
#        midx = 0 # Show a certain marker and point as text
#        pidx = 1
#        margin = 5
#        ax.text(obj_points[midx, pidx, 0] + margin, obj_points[midx, pidx, 1] + margin,
#            f'P({obj_points[midx, pidx, 0]:.1f}, {obj_points[midx, pidx, 1]:.1f}, {obj_points[midx, pidx, 2]:.1f})')
        ax.set_title('Object points')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        plt.show()

    # Load image and detect markers to get image points
    img = cv2.imread(os.path.join(data_dir, basename + '_color.png'))
    if False:
        x0_trafo = Trafo3d(t=(0, 0, 500), rpy=np.deg2rad((180, 0, 0)))
        img_ids, img_points = detect_markers(img)
    else:
        img_ids, img_points, x0_trafo = detect_markers2(img, cam, square_length,
            object_to_markers)#

    # Remove markers: Check see what happens if we remove markers
    if False:
        remove_indices = [1, 2, 3]
        img_ids = np.delete(img_ids, remove_indices)
        img_points = np.delete(img_points, remove_indices, axis=0)


    # Show image points
    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img)
        for ipnt in img_points:
            ax.plot(ipnt[:,0], ipnt[:,1], 'xr')
#        midx = 1 # Show a certain marker and point as text
#        pidx = 1
#        margin = 30
#        ax.text(img_points[midx, pidx, 0] + margin, img_points[midx, pidx, 1] + margin,
#            f'p({img_points[midx, pidx, 0]:.1f}, {img_points[midx, pidx, 1]:.1f})')
        ax.set_title('Image points')
        ax.set_xlabel('u (pixels)')
        ax.set_ylabel('v (pixels)')
        ax.set_aspect('equal')
        plt.show()

    # Get correspondences of object and image points based in IDs
    P, p = determine_correspondences(obj_ids, obj_points, img_ids, img_points)
    # Solve point-to-point problem
    cam_to_object_estim = solve_pnp(P, p, cam, x0_trafo, verbose=True)
    print(f'cam_to_object:\n    {cam_to_object}')
    print(f'cam_to_object estimated:\n    {cam_to_object_estim}')
    dt, dr = cam_to_object.distance(cam_to_object_estim)
    with np.printoptions(precision=2, suppress=True):
        print(f'Difference: {dt:.2f} mm, {np.rad2deg(dr):.2f} deg')
