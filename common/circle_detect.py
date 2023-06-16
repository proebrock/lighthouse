import numpy as np
import matplotlib.pyplot as plt
import cv2



def detect_circle_contours(image, verbose=False):
    assert image.ndim == 3 # Color image
    assert image.dtype == np.uint8 # Type uint8 [0..255]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    #thresh = cv2.adaptiveThreshold(blurred, 255,
    #        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    circles = []
    circle_contours = []
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
        circle_contours.append(np.asarray(c)[:,0,:])
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(gray, cmap='gray')
        if circles is not None:
            for circle in circles:
                ax.plot(*circle[0:2], 'r+')
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                           color='r', fill=False)
                ax.add_artist(circle_artist)
        plt.show()
    if len(circles) == 0:
        return np.zeros((0, 3)), []
    else:
        return np.array(circles), circle_contours



def detect_circle_hough(image, min_center_distance=None, min_radius=1, max_radius=500,
    verbose=False):
    assert image.ndim == 3 # Color image
    assert image.dtype == np.uint8 # Type uint8 [0..255]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
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
    if min_center_distance is None:
        minDist = rows/16
    else:
        minDist = min_center_distance
    # Higher threshold of the two passed to the Canny edge detector
    # (the lower one is twice smaller).
    param1 = 40
    # Accumulator threshold for the circle centers at the detection stage.
    # The smaller it is, the more false circles may be detected. Circles,
    # corresponding to the larger accumulator values, will be returned first.
    param2 = 30
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2,
                               minRadius=min_radius, maxRadius=max_radius)
    if circles is not None:
        circles = circles[0]
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(gray, cmap='gray')
        if circles is not None:
            for circle in circles:
                ax.plot(*circle[0:2], 'r+')
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                           color='r', fill=False)
                ax.add_artist(circle_artist)
        plt.show()
    if circles is None:
        return np.zeros((0, 3))
    else:
        return np.array(circles)
