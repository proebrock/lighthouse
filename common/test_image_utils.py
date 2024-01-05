import pytest
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from common.image_utils import image_rgb_to_gray, image_gray_to_rgb



def test_image_rgb_to_gray_dimensions_and_types():
    # Single image
    image_rgb = np.zeros((20, 32, 3), dtype=np.uint8)
    image_gray = image_rgb_to_gray(image_rgb)
    assert image_gray.shape == (20, 32)
    assert image_gray.dtype == np.uint8
    # Image stack
    images_rgb = np.zeros((10, 20, 32, 3), dtype=np.uint8)
    images_gray = image_rgb_to_gray(images_rgb)
    assert images_gray.shape == (10, 20, 32)
    assert images_gray.dtype == np.uint8



def test_image_gray_to_rgb_dimensions_and_types():
    # Single image
    image_gray = np.zeros((20, 32), dtype=np.uint8)
    image_rgb = image_gray_to_rgb(image_gray)
    assert image_rgb.shape == (20, 32, 3)
    assert image_rgb.dtype == np.uint8
    # Image stack
    images_gray = np.zeros((10, 20, 32), dtype=np.uint8)
    images_rgb = image_gray_to_rgb(images_gray)
    assert images_rgb.shape == (10, 20, 32, 3)
    assert images_rgb.dtype == np.uint8



def test_image_gray_rgb_roundtrip(random_generator):
    images_gray = random_generator.integers(0, 255,
        (10, 20, 32), dtype=np.uint8)
    images_rgb = image_gray_to_rgb(images_gray)
    images_gray2 = image_rgb_to_gray(images_rgb)
    assert np.all(images_gray == images_gray2)

