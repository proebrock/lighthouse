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
        p = np.round(circles[i,0:2]).astype(np.int)
        # Area of 5x5 pixels around circle center
        rect = image[p[1]-2:p[1]+3, p[0]-2:p[0]+3, :]
        # Average color
        color = np.mean(rect, axis=(0, 1))
        descriptors[i,:] = color
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
        kp, desc = detect_and_compute(image, verbose)
        bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        matches = bf.match(ref_desc, desc)
        if verbose:
            for m in matches:
                print(f'{m.queryIdx} -> {m.trainIdx}')
            display_image = cv2.drawMatches(ref_img, ref_kp, image, kp,
                                            matches, outImg=None, flags=2)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(display_image)
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

    # make sure every point is referenced at least once
    assert(np.unique(point_indices).size == n_points)



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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    # Load circle properties
    with open(os.path.join(data_dir, 'cam00_image00.json'), 'r') as f:
        params = json.load(f)
    sphere_centers = np.array(params['sphere']['center'])
    sphere_radius = params['sphere']['radius']

    bundle_adjust(cameras, images)