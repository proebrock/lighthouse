import copy
import cv2
import glob
import json
import matplotlib.pyplot as plt
plt.close('all')
import matplotlib.cm as cm
import numpy as np
import os
import sys

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from common.circle_detect import detect_circle_hough
from trafolib.trafo3d import Trafo3d



def detect_and_compute(image, verbose=False):
    """ Detect keypoints and features in image

    By using OpenCV data structures we are able to use matching
    methods provided by OpenCV.

    :param image: Input image
    :param verbose: Verbose (debug) output
    :return: list of n keypoints of type cv2.KeyPoint and
        descriptors of size (n x 3) of RGB colors
    """
    # Keypoints are the centers of the circles
    circles = detect_circle_hough(image, verbose)
    n = circles.shape[0]
    keypoints = []
    for i in range(n):
        kp = cv2.KeyPoint(circles[i,0], circles[i,1], circles[i,2])
        keypoints.append(kp)
    # Features are the color of the image at the center of the circle
    descriptors = np.zeros((n, 3))
    for i in range(n):
        # Circle center rounded
        p = np.round(circles[i,0:2]).astype(int)
        # Area of 5x5 pixels around circle center
        rect = image[p[1]-2:p[1]+3, p[0]-2:p[0]+3, :]
        # Average color
        color = np.mean(rect, axis=(0, 1))
        # TODO: Due to lighting problem, all spheres are too dark;
        # artificially make them brighter until problem in renderer is solved
        descriptors[i,:] = 1.1 * color
    return keypoints, descriptors.astype(np.float32)/255.0



def generate_distinct_colors20():
    steps = np.linspace(0.0, 1.0, 20)
    return cm.tab20(steps)[:,0:3]

def generate_distinct_colors40():
    steps = np.linspace(0.0, 1.0, 20)
    return np.vstack((cm.tab20b(steps), cm.tab20c(steps)))[:,0:3]

def generate_reference_descriptors():
    """ Generates keypoints and descriptors from reference colors
    This function generates an image with circle-shaped regions of
    the reference colors. The center of theses circles are the keypoints
    and the descriptors are the RGB values of the reference colors
    :return: Keypoints, descriptors and image
    """
    colors = generate_distinct_colors20()
#    colors = generate_distinct_colors40()
    radius = 25
    img = np.zeros((2*radius*len(colors), 2*radius, 3), np.uint8)
    keypoints = []
    for i, color in enumerate(colors):
        x = radius
        y = radius + 2 * i * radius
        kp = cv2.KeyPoint(x, y, radius)
        keypoints.append(kp)
        cv2.circle(img, (x, y), radius, 255*color, -1)
    return keypoints, colors.astype(np.float32), img



def obj_fun(params, cameras, n_points, camera_indices, point_indices, points_2d):
    n_cameras = len(cameras)
    camera_poses = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = np.zeros_like(points_2d)
    for i in range(n_cameras):
        mask = camera_indices == i
        cam = copy.deepcopy(cameras[i])
        cam.set_pose(Trafo3d(t=camera_poses[i,0:3], rodr=camera_poses[i,3:6]))
        points_proj[mask] = cam.scene_to_chip(points_3d[point_indices[mask]])[:,0:2]
    return (points_proj - points_2d).ravel()



def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1
    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1
    return A



def observation_to_string(index, camera_indices, point_indices):
    return f'index{camera_indices[index]:02d}.png, pnt "{point_indices[index]}"'



def bundle_adjust(cameras, images, verbose=False):
    assert(len(cameras) == len(images))
    n_cameras = len(cameras)
    print(f'n_cameras: {n_cameras}')
    ref_kp, ref_desc, ref_img = generate_reference_descriptors()
    n_points = len(ref_kp)
    print(f'n_points: {n_points}')

    # camera index an observation was made from (size: n_observations)
    camera_indices = []
    # point indices of the points the camera observed (size: n_observations)
    point_indices = []
    # 2d point of the chip coordinates of the point of point_indices
    # observed by camera of camera_indices (shape: (n_observations, 2)
    points_2d = []
    for i, image in enumerate(images):
        kp, desc = detect_and_compute(image, verbose=False)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        matches = bf.match(ref_desc, desc)
        if verbose:
            print(f'---- image{i:02d}: {len(matches)} matches -----')
            for m in matches:
                with np.printoptions(precision=2, suppress=True):
                    print(f'{m.queryIdx} ({ref_desc[m.queryIdx,:]}) -> {m.trainIdx} ({desc[m.trainIdx,:]})')
            display_image = cv2.drawMatches(ref_img, ref_kp, image, kp,
                                            matches, outImg=None, flags=0)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(display_image)
            ax.set_title(f'image{i:02d}')
            plt.show()
        for m in matches:
            camera_indices.append(i)
            point_indices.append(m.queryIdx)
            points_2d.append(kp[m.trainIdx].pt)
    camera_indices = np.array(camera_indices, dtype=int)
    point_indices = np.array(point_indices, dtype=int)
    points_2d = np.array(points_2d, dtype=float)
    n_observations = point_indices.size
    print(f'n_observations: {n_observations}')
    # Unknown are the poses (6 dim, rvec+tvec) of each camera and the 3d coordinates for each point
    n_params = 6 * n_cameras + 3 * n_points
    print(f'n_params: {n_params}')
    # Residuals are calculated based on 2d coordinates of all observations
    n_residuals = 2 * n_observations
    print(f'n_residuals: {n_residuals}')

    # Make sure every point is detected and matched at least once in any
    # of the images; so far there is no support for markers that are defined
    # but never used
    assert(np.unique(point_indices).size == n_points)

    # This method is derived from
    # "Large-scale bundle adjustment in scipy"
    # https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html
    # But we are not guessing camera parmeters but take the calibration given by parameters

    # Determine starting conditions, x0 contains camera trafos first and then 3d points
    camera_poses_0 = np.zeros((n_cameras,6))
    points_3d_0 = np.zeros((n_points,3))
    points_3d_0[:,2] = 2000.0
    x0 = np.hstack((camera_poses_0.ravel(), points_3d_0.ravel()))

    # Plot residuals of starting position
    if False:
        f0 = obj_fun(x0, cameras, n_points, camera_indices, point_indices, points_2d)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(f0)
        ax.grid()
        ax.set_ylabel('Residual (pixels)')
        plt.show()

    # Run optimizer
    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
                                   point_indices)
    res = least_squares(obj_fun, x0, jac_sparsity=A, verbose=2,
                        x_scale='jac', ftol=1e-4, method='trf',
                        args=(cameras, n_points, camera_indices,
                              point_indices, points_2d))
    if not res.success:
        raise Exception('Optimization failed: ' + str(res))

    # Plot residuals
    if False:
        def format_coord(x, y):
            index = int(round(x) / 2)
            if index >= n_observations:
                return ''
            else:
                return observation_to_string(index, camera_indices,
                    point_indices) + ': ' + str(y)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(res.fun)
        ax.format_coord = format_coord
        ax.grid()
        ax.set_ylabel('Residual (pixels)')
        plt.show()

    # Convert result
    camera_poses = res.x[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = res.x[n_cameras * 6:].reshape((n_points, 3))
    estimated_camera_poses = []
    for cp in camera_poses:
        estimated_camera_poses.append(Trafo3d(t=cp[:3], rodr=cp[3:]))
    return estimated_camera_poses, points_3d



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/bundle_adjust_large_scale'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    cameras = []
    for filename in filenames:
        with open(filename, 'r') as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cameras.append(cam)
    # Load images
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    # Load circle properties
    with open(os.path.join(data_dir, 'cam00_image00.json'), 'r') as f:
        params = json.load(f)
    sphere_centers = np.array(params['sphere']['center'])
    sphere_radius = params['sphere']['radius']

    estimated_camera_poses, estimated_sphere_centers = \
        bundle_adjust(cameras, images)

    # Real camera positions and points: Make cam0 (=step0) reference coordinate system
    ref = cameras[0].get_pose().inverse()
    camera_poses = []
    for cam in cameras:
        camera_poses.append(ref * cam.get_pose())
    sphere_centers = ref * sphere_centers
    # Est. camera positions and points: Make cam0 (=step0) reference coordinate system
    ref = estimated_camera_poses[0].inverse()
    for i in range(len(estimated_camera_poses)):
        estimated_camera_poses[i] = ref * estimated_camera_poses[i]
    estimated_sphere_centers = ref * estimated_sphere_centers

    for pose, epose in zip(camera_poses, estimated_camera_poses):
        with np.printoptions(precision=2, suppress=True):
            print('------------')
            print(f'{pose.get_translation()}, {np.rad2deg(pose.get_rotation_rpy())}')
            print(f'{epose.get_translation()}, {np.rad2deg(epose.get_rotation_rpy())}')

    with np.printoptions(precision=2, suppress=True):
        print(sphere_centers)
        print(estimated_sphere_centers)
        print(estimated_sphere_centers-sphere_centers)

