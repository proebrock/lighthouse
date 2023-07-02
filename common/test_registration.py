import numpy as np
import pytest

from trafolib.trafo3d import Trafo3d
from common.registration import estimate_transform



def test_estimate_transform_3d_without_scale(random_generator):
    # Generate points cloud
    n = 100
    p1 = random_generator.uniform(-10.0, 10.0, (n, 3))
    # Set nominal transformation parameters
    p0_to_p1 = Trafo3d(t=(10, 20, -33), rpy=np.deg2rad((-70, 120, 5)))
    # Calculate other point cloud
    p0 = p0_to_p1 * p1
    # Estimate transformation
    p0_to_p1_estim = estimate_transform(p0, p1)
    # Check results
    dt, dr = p0_to_p1.distance(p0_to_p1_estim)
    assert np.isclose(dt, 0.0)
    assert np.isclose(dr, 0.0)



def test_estimate_transform_3d_with_scale(random_generator):
    # Generate points cloud
    n = 100
    p1 = random_generator.uniform(-100.0, 100.0, (n, 3))
    # Set nominal transformation parameters
    p0_to_p1 = Trafo3d(t=(-111, 222, -303), rpy=np.deg2rad((30, -70, -145)))
    scale = 13.0
    # Calculate other point cloud
    # (with three homogenous matrices for translation (T), rotation (R) and
    # scaling (S), the total transformation would look like
    # p0 = T * R * S * p1; T * R is done in Trafo3d, so scale has to be
    # multiplied first
    p0 = p0_to_p1 * (scale * p1)
    # Estimate transformation
    p0_to_p1_estim, scale_estim = estimate_transform(p0, p1, estimate_scale=True)
    # Check results
    dt, dr = p0_to_p1.distance(p0_to_p1_estim)
    assert np.isclose(dt, 0.0)
    assert np.isclose(dr, 0.0)
    assert np.isclose(scale, scale_estim)

