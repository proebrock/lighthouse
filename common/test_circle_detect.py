import pytest
import numpy as np
import cv2

from common.circle_detect import detect_circle_contours, detect_circle_hough



@pytest.mark.parametrize('detect_func', \
    [ detect_circle_contours, detect_circle_hough ])
def test_detect_circles_nothing_to_detect(detect_func):
    # Empty test image
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    # Detection
    detected_circles, contours = detect_func(image, verbose=False)
    # Should have no results
    assert detected_circles.shape[0] == 0
    assert detected_circles.shape[1] == 3
    assert contours is None or len(contours) == 0



@pytest.mark.parametrize('detect_func', \
    [ detect_circle_contours, detect_circle_hough ])
def test_detect_circles_basic(detect_func):
    # Define test circles
    circles = np.array([
        # Row, Col, Radius
        [300,  600, 200],
        [150, 1000, 100],
        [700,  800,  50],
        [600,  300, 150],
    ])
    num_circles = circles.shape[0]
    # Draw test circles into image
    image = np.zeros((900, 1200, 3), dtype=np.uint8)
    for (row, col, radius) in circles:
        cv2.circle(image, (col, row), radius, color=(255, 0, 0),
            thickness=cv2.FILLED)
    # Detect circles
    detected_circles, contours = detect_func(image, verbose=False)
    # Check number of detected circles and contours
    assert detected_circles.shape[0] == num_circles
    if contours is not None:
        assert len(contours) == num_circles
    # Match detected circles to original circles using circle centers
    matches = -1 * np.ones(num_circles, dtype=int)
    for i in range(num_circles):
        detected_center = detected_circles[i, 0:2]
        for j in range(num_circles):
            center = circles[j, 0:2]
            if np.linalg.norm(detected_center - center) < 5.0:
                assert matches[i] == -1
                matches[i] = j
                break
    # Check results
    for i in range(num_circles):
        # Center
        detected_center = detected_circles[i, 0:2]
        center = circles[matches[i], 0:2]
        assert np.linalg.norm(detected_center - center) < 5.0
        # Radius
        detected_radius = detected_circles[i, 2]
        radius = circles[matches[i], 2]
        assert np.abs(detected_radius - radius) < 5.0
        if contours is not None:
            # Center of gravity of contour in relation to circle center
            cog = np.mean(contours[i], axis=0)
            assert np.linalg.norm(cog - center) < 5.0
            # Contour length ()= number of pixels)  in relation to circle radius
            num_pixels = contours[i].shape[0]
            assert np.abs((num_pixels / radius) - 5.65) < 0.2 # Why 5.65?

