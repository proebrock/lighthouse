# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
import pytest
import numpy as np
from . lens_distortion_model import LensDistortionModel



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def test_distort_undistort_roundtrip():
    # Generate points in a grid
    x = np.linspace(-1, 1, 11)
    y = np.linspace(-1, 1, 11)
    x, y = np.meshgrid(x, y, indexing='ij')
    p = np.zeros((x.size, 2))
    p[:,0] = x.ravel()
    p[:,1] = y.ravel()
    # Setup lens distortion model
    ldm = LensDistortionModel((-0.8, 0.8, 0.1, 0, 0, 0.2))
    # Do roundtrip
    p_dist = ldm.distort(p)
    p_undist = ldm.undistort(p_dist)
    # Calculate RMS
    residuals = p_undist - p
    residuals = np.sum(np.square(residuals), axis=1) # per point residual
    rms = np.sqrt(np.mean(np.square(residuals)))
    assert np.isclose(rms, 0)



def test_undistort_distort_roundtrip():
    # Generate points in a grid
    x = np.linspace(-1, 1, 11)
    y = np.linspace(-1, 1, 11)
    x, y = np.meshgrid(x, y, indexing='ij')
    p = np.zeros((x.size, 2))
    p[:,0] = x.ravel()
    p[:,1] = y.ravel()
    # Setup lens distortion model
    ldm = LensDistortionModel((-0.8, 0.8, 0.1, 0, 0, 0.2))
    # Do roundtrip
    p_undist = ldm.undistort(p)
    p_dist = ldm.distort(p_undist)
    # Calculate RMS
    residuals = p_dist - p
    residuals = np.sum(np.square(residuals), axis=1) # per point residual
    rms = np.sqrt(np.mean(np.square(residuals)))
    assert np.isclose(rms, 0)



if __name__ == '__main__':
    pytest.main()
