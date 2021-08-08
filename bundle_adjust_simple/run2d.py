import cv2
import itertools
import glob
import json
import numpy as np
import os
import sys
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def detect_circle_contours(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.8 * gray.size:
            # contour with 80 or more percent of total image size
            continue
        if True:
            # Result based on center of gravity and area->radius
            M = cv2.moments(c)
            circle = np.array([M["m10"] / M["m00"],
                               M["m01"] / M["m00"],
                               np.sqrt(area/np.pi)])
        else:
            # Result based on minimum enclosing circle
            circ = cv2.minEnclosingCircle(c)
            circle = np.array([circ[0][0], circ[0][1], circ[1]])
        circles.append(circle)
    if len(circles) > 0:
        circles = np.array(circles)
    else:
        circles = None
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if circles is not None:
            for circle in circles:
                ax.plot(*circle[0:2], 'r+')
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                           color='r', fill=False)
                ax.add_artist(circle_artist)
        plt.show()
    return circles



def detect_circle_hough(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    rows = blurred.shape[0]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, rows/8,
                               param1=40, param2=30,
                               minRadius=1, maxRadius=500)
    if circles is not None:
        circles = circles[0]
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if circles is not None:
            for circle in circles:
                ax.plot(*circle[0:2], 'r+')
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                           color='r', fill=False)
                ax.add_artist(circle_artist)
        plt.show()
    return circles



def bundle_adjust_objfun(x, cameras, circle_centers):
    proj_centers = np.empty_like(circle_centers)
    for i, cam in enumerate(cameras):
        p = cam.scene_to_chip(x.reshape(1, 3))
        proj_centers[i, :] = p[0, 0:2]
    return (proj_centers - circle_centers).ravel()



def compress_residuals(residuals):
    """ Compress residuals
    The bundle adjustment objective function returns
    residuals for all n cameras :
    (dx1, dy1, dx2, dy2, ..., dxn, dyn)
    This function compresses those residuals to one value per camera:
    (sqrt(dx1^2+dy1^2), sqrt(dx2^2+dy2^2), ..., sqrt(dxn^2+dyn^2))
    :param residuals: Residuals from bundle_adjust_objfun
    :return: Residuals reduced to one value per camera
    """
    res_sq = np.square(residuals)
    res_sq_per_cam = np.array(list(res_sq[2*i] + res_sq[2*i+1] for i in range(res_sq.size//2)))
    return np.sqrt(res_sq_per_cam)



def estimate_error(sphere_center, cameras, circle_centers):
    """ Estimate bundle adjustment error in world coordinates

    In bundle-adjustment, a natural way to describe the residual
    error of the estimation is the re-projection error. It comes as a
    natural result of the numerical optimization. Unfortunately it is
    defined in camera pixels and that is not helpful to evaluate the
    error in terms of world coordinates.

    This method determines the error as the distance between the
    camera rays (ray of detected circle center into the scene) and
    the final solution of the bundle adjustment.

    :param sphere_center: Solution of bundle adjustment
    :param cameras: List of cameras
    :param circle_centers: List of circle centers (position of detected object)
    :return: Distances of camera rays to sphere_center
    """
    x0 = sphere_center
    errors = np.empty(len(cameras))
    for i, (cam, center) in enumerate(zip(cameras, circle_centers)):
        # Get two points x1 and x2 on the camera ray
        x1 = cam.get_camera_pose().get_translation()
        p = np.array([[center[0], center[1], 100]])
        x2 = cam.chip_to_scene(p)
        # Get distance of sphere_center (x0) to camera ray (x1, x2)
        errors[i] = np.linalg.norm(np.cross(x0-x1, x0-x2))/np.linalg.norm(x2-x1)
    return errors



def bundle_adjust(cameras, circle_centers, x0=np.array([0, 0, 100])):
    res = least_squares(bundle_adjust_objfun, x0, xtol=0.1,
                        args=(cameras, circle_centers))
    if not res.success:
        raise Exception(f'Bundle adjustment failed: {res}')
    sphere_center = res.x
    residuals = compress_residuals(res.fun)
    errors = estimate_error(res.x, cameras, circle_centers)
    return sphere_center, residuals, errors



def bundle_adjust_sac(cameras, circle_centers, threshold, verbose=False):
    n = len(cameras)
    # This is similar to a RANSAC approach: In RANSAC you would
    # take random pairs cameras to calculate the solution with (two
    # cameras are the minimum number of cameras to find a solution);
    # since the total number of cameras is usually small, we use
    # all combinations of two cameras from the list of cameras;
    # number of combinations (no repetitions) is
    # n!/(2*(n-2)!) = n*(n-1)/2; so it is O(n^2)
    combs = np.array(list(itertools.combinations(np.arange(n), 2)))
    residuals = np.empty((len(combs), n))
    for i, (j, k) in enumerate(combs):
        # Run bundle adjustment with two cameras (min number of cams
        # for a solution) taken from combs
        cams = (cameras[j], cameras[k])
        ccenters = (circle_centers[(j, k), :])
        x, _, _ = bundle_adjust(cams, ccenters)
        # Use this solution from two cameras and determine residuals for ALL cams
        fun = bundle_adjust_objfun(x, cameras, circle_centers)
        residuals[i, :] = compress_residuals(fun)
    # For each solution from combs determine number of inlier cameras
    # by checking residuals against a maximum theshold (in pixels)
    all_inliers = residuals < threshold
    # Use row with maximum number of inliers
    max_inliers_index = np.argmax(np.sum(all_inliers, axis=1))
    # This is a bool vector determining the inliers of cameras
    cam_inliers = all_inliers[max_inliers_index, :]
    if verbose:
        with np.printoptions(precision=1, suppress=True):
            print('combs\n', combs)
            print('residuals\n', residuals)
            print('all_inliers\n', all_inliers)
            print('max_inliers_index\n', max_inliers_index)
            print('cam_inliers\n', cam_inliers)
    # Run bundle adjustment with all inlier cameras to determine result
    good_cams = list(cameras[i] for i in np.where(cam_inliers)[0])
    good_circle_centers = circle_centers[cam_inliers, :]
    x, _, _ = bundle_adjust(good_cams, good_circle_centers)
    # Use this solution from inlier cameras and determine residuals for ALL cams
    fun = bundle_adjust_objfun(x, cameras, circle_centers)
    residuals = compress_residuals(fun)
    errors = estimate_error(x, cameras, circle_centers)
    return x, residuals, errors, cam_inliers



def visualize_scene(cameras, circle_centers):
    scene = []
    for cam, center in zip(cameras, circle_centers):
        scene.append(cam.get_cs(size=100.0))
        #scene.append(cam.get_frustum(size=200.0))
        line_set = o3d.geometry.LineSet()
        points = np.empty((2, 3))
        points[0, :] = cam.get_camera_pose().get_translation()
        p = np.array((center[0], center[1], 1200)).reshape(1, 3)
        P = cam.chip_to_scene(p)
        points[1, :] = P[0, :]
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [[0, 0, 0]]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        scene.append(line_set)
    o3d.visualization.draw_geometries(scene)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    # Config
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/bundle_adjust_simple'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    cameras = []
    sphere_radius = None
    for filename in filenames:
        print(f'Loading camera from {filename} ...')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cameras.append(cam)
        sphere_center = np.array(params['sphere']['center'])
        sphere_radius = params['sphere']['radius']

    # Load images
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images = []
    for filename in filenames:
        print(f'Loading image from {filename} ...')
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    assert len(cameras) == len(images) # Assumption: each camera takes one image of sphere

    # Detect center of circles in images
    circle_centers = np.empty((len(images), 2))
    for i, img in enumerate(images):
        print(f'Detecting sphere in image {i+1}/{len(images)} ...')
        circ = detect_circle_contours(img, verbose=False)
#        circ = detect_circle_hough(img, verbose=False)
        if circ is None or circ.shape[0] != 1:
            raise Exception(f'Found more than one circle in image {i}')
        circle_centers[i, :] = circ[0, 0:2]
    visualize_scene(cameras, circle_centers)

    if False:
        fig = plt.figure()
        for i in range(len(images)):
            ax = fig.add_subplot(2, 2, i+1)
            ax.imshow(images[i])
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.set_title(f'Cam{i}')
            x, y = circle_centers[i,0], circle_centers[i,1]
            ax.plot(x, y, '+y')
            margin = 75
            ax.text(x + margin, y + margin, f'({x:.1f},{y:.1f})', color='k')
            ax.set_aspect('equal')
        plt.show()

    # Run bundle adjustment
    print('\nRunning bundle adjustment ...')
    estimated_sphere_center, residuals, errors = bundle_adjust(cameras, circle_centers)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Reprojection error per cam {residuals} pix')
        print(f'Errors per cam {errors} mm')

    # Add small error to camera pose of one camera
    T_small = Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 1, 0)))
    cam_no = 1
    cameras[cam_no].set_camera_pose(cameras[cam_no].get_camera_pose() * T_small)
    visualize_scene(cameras, circle_centers)

    # Re-run bundle adjustment
    print(f'\nRe-running bundle adjustment after misaligning camera {cam_no}...')
    estimated_sphere_center, residuals, errors = bundle_adjust(cameras, circle_centers)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Reprojection error per cam {residuals} pix')
        print(f'Errors per cam {errors} mm')

    # Run bundle adjustment with sample consensus (SAC) approach
    threshold = 3.0
    print(f'\nRe-running SAC bundle adjustment (threshold={threshold} pix) after misaligning camera {cam_no}...')
    estimated_sphere_center, residuals, errors, cam_inliers = \
        bundle_adjust_sac(cameras, circle_centers, threshold, verbose=False)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Reprojection error per cam {residuals} pix')
        print(f'Errors per cam {errors} mm')
        print(f'Inlier cameras: {cam_inliers}')
