# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
import random as rand
from cameraModel import CameraModel



@pytest.fixture
def random():
    rand.seed(0)
    numpy.random.seed(0)



def test_TrafoChipToSceneAndBack():
    # Generate test points on chip
    width = 640
    height = 480
    n = 100
    p = np.hstack((
        (2.0 * width * np.random.rand(n,1) - width) / 2.0,
        (2.0 * height * np.random.rand(n,1) - height) / 2.0,
        1000.0 * np.random.rand(n,1) + 1))
    # Transform to scene and back
    cam = CameraModel((width, height), 50)
    P = cam.chipToScene(p)
    p2 = cam.sceneToChip(P)
    # Should still be the same
    assert(np.allclose(p - p2, 0.0))



def test_TrafoDepthImageToSceneAndBack():
    # Generate test depth image
    width = 640
    height = 480
    img = 1000.0 * np.random.rand(width, height)
    # Set every 10th pixel to NaN
    num_nan = (width * height) // 10
    nan_idx = np.hstack(( \
        np.random.randint(0, width, size=(num_nan, 1)), \
        np.random.randint(0, height, size=(num_nan, 1))))
    img[nan_idx[:,0], nan_idx[:,1]] = np.nan
    # Transform to scene and back
    cam = CameraModel((width, height), 50)
    P = cam.depthImageToScenePoints(img)
    img2 = cam.scenePointsToDepthImage(P)
    # Should still be the same
    assert np.all(np.isnan(img) == np.isnan(img2))
    mask = ~np.isnan(img)
    assert np.allclose(img[mask], img2[mask])

