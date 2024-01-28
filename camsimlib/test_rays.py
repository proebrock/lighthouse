import pytest
import numpy as np

from . rays import Rays



def test_invalid_origs_and_dirs():
    rayorigs = np.zeros((2, 3))
    raydirs = np.zeros((5, 3))
    with pytest.raises(ValueError):
        rays = Rays(rayorigs, raydirs)



def test_to_points_distances():
    rayorigs = np.array((
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        (0, 0, 0),
        ))
    raydirs = np.array((
        (1, 0, 0),
        (1, 0, 0),
        (0, 2, 0),
        (0, 0, 3),
        ))
    points = np.array((
        (0, 0, 0), # at ray orig
        (1, 0, 0), # at ray orig + ray dir
        (1, 0, 0), # besides ray in forward direction
        (0, 2, -3), # besides ray in backwards direction
        ))
    expected_distances = np.array((0, 0, 1, 2))
    rays = Rays(rayorigs, raydirs)
    distances = rays.to_points_distances(points)
    assert np.all(np.isclose(distances, expected_distances))

    # Move rays AND points, expect the same result!
    delta = np.array((1, -2, 3))
    rayorigs += delta
    points += delta
    distances = rays.to_points_distances(points)
    assert np.all(np.isclose(distances, expected_distances))



def test_intersect_with_plane_cornercases():
    # Generate plane: x/y plane
    plane = np.array((0.0, 0.0, 1.0, 0.0))
    # Generate rays
    ray_origs = np.array((
        (1.0, -1.0, 0.0), # on plane
        (1.0, -1.0, 0.0),
        (1.0, -1.0, 0.0),
        (1.0, -1.0, 1.0), # above plane
        (1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, -1.0, -1.0), # under plane
        (1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
    ))
    ray_dirs = np.array((
        (0.0, 0.0, 1.0), # upwards
        (0.0, 1.0, 0.0), # orthogonally to plane
        (0.0, 0.0, -1.0), # downwards
        (0.0, 0.0, 1.0), # upwards
        (0.0, 1.0, 0.0), # orthogonally to plane
        (0.0, 0.0, -1.0), # downwards
        (0.0, 0.0, 1.0), # upwards
        (0.0, 1.0, 0.0), # orthogonally to plane
        (0.0, 0.0, -1.0), # downwards
    ))
    rays = Rays(ray_origs, ray_dirs)
    # Intersect
    points, mask, scales = rays.intersect_with_plane(plane)
    # Check results
    expected_mask = np.array((
        True, False, True,
        True, False, True,
        True, False, True,
    ), dtype=bool)
    assert np.all(mask == expected_mask)
    expected_scales = np.array((0.0, 0.0, -1.0, 1.0, 1.0, -1.0))
    assert np.allclose(scales, expected_scales)
    expected_points = np.array((
        (1., -1., 0.),
        (1., -1., 0.),
        (1., -1., 0.),
        (1., -1., 0.),
        (1., -1., 0.),
        (1., -1., 0.),
    ))
    assert np.allclose(points, expected_points)
