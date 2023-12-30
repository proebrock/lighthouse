import pytest

import numpy as np
from . lens_distortion_model import LensDistortionModel
import matplotlib.pyplot as plt



def generate_point_grid(xmin, xmax, numx, ymin, ymax, numy):
    x = np.linspace(xmin, xmax, numx)
    y = np.linspace(ymin, ymax, numy)
    x, y = np.meshgrid(x, y, indexing='ij')
    p = np.zeros((x.size, 2))
    p[:,0] = x.ravel()
    p[:,1] = y.ravel()
    return p



def test_distort_undistort_roundtrip():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model
    ldm = LensDistortionModel((-0.8, 0.8, 0.1, 0, 0.2))
    # Do roundtrip
    p_dist = ldm.distort(p)
    p_undist = ldm.undistort(p_dist)
    # Check
    assert np.allclose(p, p_undist)



def test_undistort_distort_roundtrip():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model
    ldm = LensDistortionModel((-0.8, 0.8, 0.1, 0, 0.2))
    # Do roundtrip
    p_undist = ldm.undistort(p)
    p_dist = ldm.distort(p_undist)
    # Check
    assert np.allclose(p, p_dist)



def test_no_distortion():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model with all distortion coefficients zero
    ldm = LensDistortionModel((0, 0, 0, 0, 0))
    # Distorting and undistorting should change nothing
    p_undist = ldm.undistort(p)
    assert np.allclose(p, p_undist)
    p_dist = ldm.distort(p)
    assert np.allclose(p, p_dist)



def test_radial_undistort():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model
    k1 = -0.8
    k2 = 0.8
    k3 = 0.1
    k4 = 0.02
    k5 = 0.0
    k6 = 0.06
    ldm = LensDistortionModel((k1, k2, 0, 0, k3, k4, k5, k6))
    p_undist = ldm.undistort(p)
    # Manually calculate undistort and compare results
    rsq = p[:,0] * p[:,0] + p[:,1] * p[:,1]
    t = (1.0 + k1 * rsq + k2 * rsq**2 + k3 * rsq**3) /\
        (1.0 + k4 * rsq + k5 * rsq**2 + k6 * rsq**3)
    p_undist2 = t[:,np.newaxis] * p
    assert np.allclose(p_undist, p_undist2)



def test_tangential_undistort():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model
    p1 = 0.05
    p2 = -0.02
    ldm = LensDistortionModel((0, 0, p1, p2, 0))
    p_undist = ldm.undistort(p)
    # Manually calculate undistort and compare results
    rsq = p[:,0] * p[:,0] + p[:,1] * p[:,1]
    p_undist2 = np.empty_like(p)
    p_undist2[:,0] = p[:,0] + 2*p1*p[:,0]*p[:,1] + p2*(rsq+2*p[:,0]**2)
    p_undist2[:,1] = p[:,1] + p1*(rsq+2*p[:,1]**2) + 2*p2*p[:,0]*p[:,1]
    assert np.allclose(p_undist, p_undist2)



def test_thin_prism_undistort():
    # Generate points in a grid
    p = generate_point_grid(-1, 1, 21, -1, 1, 21)
    # Setup lens distortion model
    s1 = 0.01
    s2 = -0.02
    s3 = -0.03
    s4 = 0.04
    ldm = LensDistortionModel((0, 0, 0, 0, 0, 0, 0, 0, s1, s2, s3, s4))
    p_undist = ldm.undistort(p)
    # Manually calculate undistort and compare results
    rsq = p[:,0] * p[:,0] + p[:,1] * p[:,1]
    p_undist2 = np.empty_like(p)
    p_undist2[:,0] = p[:,0] + s1*rsq + s2*rsq**2
    p_undist2[:,1] = p[:,1] + s3*rsq + s4*rsq**2
    assert np.allclose(p_undist, p_undist2)



def test_radial_distort():
    # Generate points in a grid
    p = generate_point_grid(-0.5, 0.5, 21, -0.5, 0.5, 21)
    # Setup lens distortion model
    k1 = -0.1
    k2 = 0.1
    k3 = 0.2
    k4 = 0.08
    ldm = LensDistortionModel((k1, k2, 0, 0, k3, k4))
    p_dist = ldm.distort(p)
    # Manually calculate distort and compare results
    # Parameters taken from Pierre Drap: "An Exact Formula
    # for Calculating Inverse Radial Lens Distortions" 2016
    b = np.array([
        -k1,
        3*k1**2 - k2,
        -12*k1**3 + 8*k1*k2 - k3,
        55*k1**4 - 55*k1**2*k2 + 5*k2**2 + 10*k1*k3 - k4,
        -273*k1**5 + 364*k1**3*k2 - 78*k1*k2**2 - 78*k1**2*k3 +
        12*k2*k3 + 12*k1*k4,
        1428*k1**6 - 2380*k1**4*k2 + 840*k1**2*k2**2 - 35*k2**3 +
        560*k1**3*k3 -210*k1*k2*k3 + 7*k3**2 - 105*k1**2*k4 + 14*k2*k4,
        -7752*k1**7 + 15504*k1**5*k2 - 7752*k1**3*k2**2 +
        816*k1*k2**3 - 3876*k1**4*k3 + 2448*k1**2*k2*k3 - 136*k2**2*k3 -
        136*k1*k3**2 + 816*k1**3*k4 - 272*k1*k2*k4 + 16*k3*k4,
        43263*k1**8 - 100947*k1**6*k2 + 65835*k1**4*k2**2 -
        11970*k1**2*k2**3 + 285*k2**4 + 26334*k1**5*k3 -
        23940*k1**3*k2*k3 + 3420*k1*k2**2*k3 + 1710*k1**2*k3**2 -
        171*k2*k3**2 - 5985*k1**4*k4 + 3420*k1**2*k2*k4 - 171*k2**2*k4 -
        342*k1*k3*k4 + 9*k4**2,
        -246675*k1**9 + 657800*k1**7*k2 - 531300*k1**5*k2**2 +
        141680*k1**3*k2**3 - 8855*k1*k2**4 - 177100*k1**6*k3 +
        212520*k1**4*k2*k3 - 53130*k1**2*k2**2*k3 + 1540*k2**3*k3 -
        17710*k1**3*k3**2 + 4620*k1*k2*k3**2 - 70*k3**3 + 42504*k1**5*k4 -
        35420*k1**3*k2*k4 + 4620*k1*k2**2*k4 + 4620*k1**2*k3*k4 -
        420*k2*k3*k4 - 210*k1*k4**2
    ])
    ssq = p[:,0] * p[:,0] + p[:,1] * p[:,1]
    ssqvec = np.array(list(ssq**(i+1) for i in range(b.size)))
    t = 1.0 + np.dot(b, ssqvec)
    p_dist2 = t[:,np.newaxis] * p

    assert np.allclose(p_dist, p_dist2, atol=0.1)
