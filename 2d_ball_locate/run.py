import cv2
import glob
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
plt.close('all')

sys.path.append(os.path.abspath('../'))
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



def estimate_sphere_position(cam, circle_center, circle_radius, sphere_radius):
    f = np.mean(cam.get_focal_length())
    p = np.array([[
            circle_center[0], circle_center[1],
            (f * sphere_radius) / circle_radius
            ]])
    return cam.chip_to_scene(p)



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'b'
    #data_dir = '/home/phil/pCloudSync/data/leafstring/2d_ball_locate'
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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        # TODO: Maybe un-distort images first? Does not help against fx!=fy
        circ = detect_circle_contours(img, False)
#        circ = detect_circle_hough(img, False)
        if circ is None or circ.shape[0] != 1:
            raise Exception(f'Found more than one circle in image {i}')
        circles[i,:] = circ[0,:]

    # Reconstruct sphere position from circle parameters
    estimated_sphere_centers = np.zeros((len(images), 3))
    for i in range(len(images)):
        estimated_sphere_centers[i,:] = estimate_sphere_position( \
            cam, circles[i,0:2], circles[i,2], sphere_radius)

    # Analysis
    print('Ranges of real sphere centers')
    print(np.min(sphere_centers, axis=0))
    print(np.max(sphere_centers, axis=0))
    errors = estimated_sphere_centers - sphere_centers
    abs_errors = np.linalg.norm(errors, axis=1)

    fig, ax = plt.subplots()
    ax.set_title('Error per coordinate')
    ax.boxplot(errors)
    ax.set_xticklabels(['X', 'Y', 'Z'])
    ax.set_xlabel('Coordinate')
    ax.set_ylabel('Distance of estimated and real 3d sphere positions (mm)')
    ax.yaxis.grid(True)

    max_abs_error_index = np.argmax(abs_errors)
    print(f'Max absolute error: {errors[max_abs_error_index,:]}')
    print(f'  Real position: {sphere_centers[max_abs_error_index,:]}')
    print(f'  Estimated position: {estimated_sphere_centers[max_abs_error_index,:]}')
    circ = detect_circle_contours(images[max_abs_error_index], True)

    sphere_dist = np.linalg.norm(sphere_centers, axis=1)
    fig, ax = plt.subplots()
    ax.plot(sphere_dist, abs_errors, 'o')
    ax.grid()
    ax.set_xlabel('Real sphere distance from camera (mm)')
    ax.set_ylabel('Distance of estimated and real 3d sphere positions (mm)')
    plt.show()