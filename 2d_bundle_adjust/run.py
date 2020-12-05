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
    if res.success:
        return res.x
    else:
        return np.NaN * np.zeros(3)



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
print('Running bundle adjustment')
estimated_sphere_center = bundle_adjust(cameras, circle_centers)
print(f'Real sphere center at {sphere_center}')
print(f'Estimated sphere center at {estimated_sphere_center}')
print(f'Error {estimated_sphere_center - sphere_center}')
