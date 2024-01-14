import pytest

import numpy as np

from . image_mapping import image_points_to_indices, image_indices_to_points,\
    image_points_on_chip_mask, image_indices_on_chip_mask



def test_image_points_image_indices_roundtrip(random_generator):
    n = 100
    points = random_generator.uniform(-100.0, 100, (n, 2))
    indices = image_points_to_indices(points)
    points2 = image_indices_to_points(indices)
    assert np.all(np.isclose(points, points2))



def test_image_points_image_indices_validity(random_generator):
    n = 100
    points = random_generator.uniform(-10.0, 60, (n, 2))
    indices = image_points_to_indices(points)
    chip_size = (50, 50)
    points_mask = image_points_on_chip_mask(points, chip_size)
    indices_mask = image_indices_on_chip_mask(indices, chip_size)
    assert np.all(points_mask == indices_mask)
