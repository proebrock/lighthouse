import copy
import cv2
import json
import numpy as np
import os
import sys
import open3d as o3d
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel



def load_files(data_dir, num_cams, num_imgs, verbose=False):
    cameras = []
    images = []
    sphere_centers = np.empty((num_imgs, 3))
    sphere_radius = None
    times = np.empty(num_imgs)
    for img_no in range(num_imgs):
        images_per_cam = []
        for cam_no in range(num_cams):
            basename = os.path.join(data_dir, f'cam{cam_no:02d}_image{img_no:02d}')
            if verbose:
                print(f'Loading from {basename} ...')
            with open(basename + '.json', 'r') as f:
                params = json.load(f)
            if cam_no == 0:
                sphere_centers[img_no,:] = params['sphere']['center']
                sphere_radius = params['sphere']['radius']
                times[img_no] = params['time']
            if img_no == 0:
                cam = CameraModel()
                cam.dict_load(params['cam'])
                cameras.append(cam)
            img = cv2.imread(basename + '_color.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images_per_cam.append(img)
        images.append(images_per_cam)
    return cameras, images, times, sphere_centers, sphere_radius



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
            print(f'cam{i}: {cam.get_camera_pose()}')
        objs.append(cam.get_cs(size=100.0))
        objs.append(cam.get_frustum(size=500.0))
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    # Config
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/bundle_adjust_trajectory'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')
    num_cams = 4
    num_imgs = 42
    cameras, images, times, sphere_centers, sphere_radius = \
        load_files(data_dir, num_cams, num_imgs)

    tic = time.monotonic()
    estimated_sphere_centers = np.zeros((num_imgs, 3))
    estimated_errors = np.zeros((num_imgs, num_cams))
    estimated_errors[:] = np.NaN
    for img_no, images_per_cam in enumerate(images):
        # images_per_cam contains the images of all camera from
        # the current step of the trajectory: we detect the circle
        # centers for all cameras if available
        cam_indices = []
        cams = []
        circle_centers = []
        for cam_no, img in enumerate(images_per_cam):
            circ = detect_circle_contours(img, verbose=False)
            if circ is not None and circ.shape[0] == 1:
                cam_indices.append(cam_no)
                cams.append(cameras[cam_no])
                circle_centers.append(circ[0,0:2])
        # Check if enough data to reconstruct
        if len(cams) < 2:
            raise Exception('At least two valid camera images needed to reconstruct trajectory step.')
        # Get a good estimate x0 of the next position of the sphere
        # based on the previous estimations as starting point for optimization
        if img_no == 0:
            # No previous data: just pick a number in front of the camera
            x0 = np.array([0, 0, 500])
        else:
            # Take last known position
            x0 = estimated_sphere_centers[img_no-1,:]
            if img_no >= 2:
                # Correct for current speed
                s1 = estimated_sphere_centers[img_no-1,:]
                s2 = estimated_sphere_centers[img_no-2,:]
                v = s1 - s2
                x0 = x0 + v
            if img_no >= 3:
                # Correct for acceleration
                s1 = estimated_sphere_centers[img_no-1,:]
                s2 = estimated_sphere_centers[img_no-2,:]
                s3 = estimated_sphere_centers[img_no-3,:]
                v1 = s1 - s2
                v2 = s2 - s3
                a = (v1 - v2)
                x0 = x0 + a
        # Do bundle adjustment for current step of trajectory
        sc, res, err = bundle_adjust(cams, circle_centers, x0)
        estimated_sphere_centers[img_no,:] = sc
        for i in range(len(cam_indices)):
            estimated_errors[img_no, cam_indices[i]] = err[i]
    toc = time.monotonic()
    print(f'Reconstructing trajectory took {(toc - tic):.1f}s')

    max_error_dist = np.linalg.norm(estimated_sphere_centers - sphere_centers, axis=1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(max_error_dist, label='Real error')
    for cam_no in range(num_cams):
        ax.plot(estimated_errors[:,cam_no], label=f'Estimated error cam{cam_no}')
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Error (mm)')
    ax.grid()
    ax.legend()
    plt.show()

    # Create sphere (for visualization´)
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.paint_uniform_color((0.2, 0.3, 0.4))

    visualize_scene(sphere, estimated_sphere_centers, cameras)
