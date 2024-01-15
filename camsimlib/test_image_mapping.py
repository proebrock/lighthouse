import pytest

import numpy as np

from . image_mapping import image_points_to_indices, image_indices_to_points,\
    image_points_on_chip_mask, image_indices_on_chip_mask, \
    image_sample_points_coarse



def test_image_points_image_indices_roundtrip(random_generator):
    n = 100
    points = random_generator.uniform(-100.0, 100, (n, 2))
    indices = image_points_to_indices(points)
    points2 = image_indices_to_points(indices)
    assert np.all(np.isclose(points, points2))



def test_image_points_image_indices_validity_random(random_generator):
    n = 100
    points = random_generator.uniform(-10.0, 60, (n, 2))
    indices = image_points_to_indices(points)
    chip_size = (50, 50)
    points_mask = image_points_on_chip_mask(points, chip_size)
    indices_mask = image_indices_on_chip_mask(indices, chip_size)
    assert np.all(points_mask == indices_mask)



def test_image_points_image_indices_validity_corner_cases(random_generator):
    EPS = 1e-6
    points = np.array([
        [ 0.0, 0.0 ],
        [ 0.0-EPS, 0.0-EPS ],
        [ 0.0+EPS, 0.0+EPS ],
        [ 3.0, 2.0 ],
        [ 3.0-EPS, 2.0-EPS ],
        [ 3.0+EPS, 2.0+EPS ],
    ])
    indices = image_points_to_indices(points)
    chip_size = (3, 2)
    points_mask = image_points_on_chip_mask(points, chip_size)
    indices_mask = image_indices_on_chip_mask(indices, chip_size)
    expected_on_chip = [ True, False, True, False, True, False ]
    assert np.all(expected_on_chip == points_mask)
    assert np.all(expected_on_chip == indices_mask)



def test_image_sample_points_coarse_manual_points():
    image = np.zeros((2, 3, 3))
    image[:, :, 0] = np.arange(6).reshape((2, 3)) # Red channel
    EPS = 1e-6
    points = np.array([
        [ 1.0-EPS, 1.5 ],
        #[ 1.0, 1.5 ],
        [ 1.0+EPS, 1.5 ],
        [ 1.5, 1.5 ],
        [ 2.0-EPS, 1.5 ],
        #[ 2.0, 1.5 ],
        [ 2.0+EPS, 1.5 ],
    ])
    values, _ = image_sample_points_coarse(image, points)
    values = values[:, 0] # Red channel
    expected_values = [ 3.0, 4.0, 4.0, 4.0, 5.0 ]
    assert np.all(np.isclose(values, expected_values))
