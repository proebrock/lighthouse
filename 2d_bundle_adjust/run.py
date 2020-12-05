import cv2
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




def detect_circle(img, verbose=False):
    display_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(display_img, contours, -1, (0, 0, 255), 3)
    areas = []
    centers = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.8 * gray.size:
            # contour with 80 or more percent of total image size
            continue
        areas.append(area)
        M = cv2.moments(c)
        center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        centers.append(center)
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
        for area, center in zip(areas, centers):
            ax.plot(center[0], center[1], 'r+')
        plt.show()
    assert(len(centers) == 1) # Assumption: Just one circle in image
    return centers[0]



def bundle_adjust_objfun(x, cameras, circle_centers):
    proj_centers = np.empty_like(circle_centers)
    for i, cam in enumerate(cameras):
        p = cam.scene_to_chip(x.reshape(1,3))
        proj_centers[i, :] = p[0, 0:2]
    return (proj_centers - circle_centers).ravel()



def bundle_adjust(cameras, circle_centers):
    x0 = np.array([0, 0, 100])
    res = least_squares(bundle_adjust_objfun, x0, xtol=0.1,
                        args=(cameras, circle_centers))
    if not res.success:
        raise Exception(f'Bundle adjustment failed: {res}')
    res_sq = np.square(res.fun)
    res_sq_per_cam = np.array(list(res_sq[2*i] + res_sq[2*i+1] for i in range(res_sq.size//2)))
    return res.x, np.sqrt(res_sq_per_cam)



def visualize_scene(cameras, circle_centers):
    scene = []
    for cam, center in zip(cameras, circle_centers):
        scene.append(cam.get_cs(size=100.0))
        line_set = o3d.geometry.LineSet()
        points = np.empty((2,3))
        points[0,:] = cam.get_camera_pose().get_translation()
        p = np.array((center[0], center[1], 1200)).reshape(1, 3)
        P = cam.chip_to_scene(p)
        points[1,:] = P[0,:]
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = [[0, 0, 0]]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        scene.append(line_set)
    o3d.visualization.draw_geometries(scene)


# Config
data_dir = 'a'
#data_dir = '/home/phil/pCloudSync/data/leafstring/2d_bundle_adjust'
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
    images.append(img)

assert(len(cameras) == len(images)) # Assumption: each camera takes one image of sphere

# Detect center of circles in images
circle_centers = np.empty((len(images), 2))
for i, img in enumerate(images):
    print(f'Detecting sphere in image {i+1}/{len(images)} ...')
    center = detect_circle(img, verbose=False)
    circle_centers[i,:] = center

# Run bundle adjustment
print('\nRunning bundle adjustment ...')
estimated_sphere_center, residuals = bundle_adjust(cameras, circle_centers)
print(f'Real sphere center at {sphere_center}')
print(f'Estimated sphere center at {estimated_sphere_center}')
print(f'Error {estimated_sphere_center - sphere_center}')
with np.printoptions(precision=2):
    print(f'Residuals per cam {residuals} pix')
#visualize_scene(cameras, circle_centers)

# Add small error to camera pose of one camera
T_small = Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 1, 0)))
cam_no = 1
cameras[cam_no].set_camera_pose(cameras[cam_no].get_camera_pose() * T_small)
print(f'\nRe-running bundle adjustment after misaligning camera {cam_no}...')
estimated_sphere_center, residuals = bundle_adjust(cameras, circle_centers)
print(f'Real sphere center at {sphere_center}')
print(f'Estimated sphere center at {estimated_sphere_center}')
print(f'Error {estimated_sphere_center - sphere_center}')
with np.printoptions(precision=2):
    print(f'Residuals per cam {residuals} pix')
visualize_scene(cameras, circle_centers)
