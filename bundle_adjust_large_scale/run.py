import cv2
import glob
import json
import matplotlib.pyplot as plt
plt.close('all')
import matplotlib.cm as cm
import numpy as np
import os
import sys



sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel



def detect_circle_hough(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    rows = blurred.shape[0]
    # Inverse ratio of the accumulator resolution to the image resolution.
    # For example, if dp=1 , the accumulator has the same resolution as
    # the input image. If dp=2 , the accumulator has half as big width
    # and height.
    dp = 1
    # Minimum distance between the centers of the detected circles.
    # If the parameter is too small, multiple neighbor circles may
    # be falsely detected in addition to a true one. If it is too large,
    # some circles may be missed.
    minDist = rows/16
    # Higher threshold of the two passed to the Canny edge detector
    # (the lower one is twice smaller).
    param1 = 40
    # Accumulator threshold for the circle centers at the detection stage.
    # The smaller it is, the more false circles may be detected. Circles,
    # corresponding to the larger accumulator values, will be returned first.
    param2 = 30
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
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



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/leafstring/bundle_adjust_large_scale'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    cameras = []
    for filename in filenames:
        with open(os.path.join(data_dir, 'cam00_image00.json'), 'r') as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cameras.append(cam)
    # Load images
    filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images = []
    for filename in filenames:
        img = cv2.imread(filename)
        images.append(img)
    # Load circle properties
    with open(os.path.join(data_dir, 'cam00_image00.json'), 'r') as f:
        params = json.load(f)
    sphere_centers = np.array(params['sphere']['center'])
    sphere_radius = params['sphere']['radius']

