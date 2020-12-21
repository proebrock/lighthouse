import copy
import cv2
import glob
import json
import numpy as np
import os
import sys
import open3d as o3d
import time
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



def detect_circle_hough(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    rows = blurred.shape[0]
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, rows/8,
                               param1=40, param2=30,
                               minRadius=1, maxRadius=500)
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if circles is not None:
            for circle in circles[0]:
                ax.plot(*circle[0:2], 'r+')
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                           color='r', fill=False)
                ax.add_artist(circle_artist)
        plt.show()
    if circles is not None:
        return circles[0]
    else:
        return None



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/leafstring/bundle_adjust_trajectory'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load camera
    with open(os.path.join(data_dir, 'image00.json'), 'r') as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    # Load images
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        images.append(img)
    # Load real sphere positions
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    sphere_centers = np.zeros((len(images), 3))
    sphere_radius = None
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            params = json.load(f)
        sphere_centers[i,:] = params['sphere']['center']
        sphere_radius = params['sphere']['radius']

    # Run circle detection on images
    circles = np.zeros((len(images), 3))
    for i, img in enumerate(images):
        circ = detect_circle_hough(img, False)
        if circ is None or circ.shape[0] != 1:
            circ = detect_circle_hough(img, True)
            raise Exception(f'Found more than one circle in image {i}')
        circles[i,:] = circ[0,:]
