import pytest
import numpy as np

from . rays import Rays



def test_invalid_origs_and_dirs():
    rayorigs = np.zeros((2, 3))
    raydirs = np.zeros((5, 3))
    with pytest.raises(ValueError):
        rays = Rays(rayorigs, raydirs)



def test_distances():
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
