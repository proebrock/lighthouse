import pytest

import numpy as np

from . image_mapping import image_points_to_indices, image_indices_to_points,\
    image_points_on_chip_mask, image_indices_on_chip_mask, \
    image_sample_points_nearest, image_sample_points_bilinear



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



def test_image_points_image_indices_validity_corner_cases():
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



#########################################



@pytest.mark.parametrize('sample_func', \
    [ image_sample_points_nearest, image_sample_points_bilinear ])
def test_image_sample_points_rgb(sample_func):
    """ Check if sampling works with RGB images
    """
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(6).reshape((2, 3)) # R
    image[:, :, 1] = np.arange(6).reshape((2, 3)) # G
    image[:, :, 2] = np.arange(6).reshape((2, 3)) # B
    points = np.array([[2.5, 1.5]])
    samples, _ = sample_func(image, points)
    assert np.all(samples == [5.0, 5.0, 5.0])
    assert samples.dtype == image.dtype



@pytest.mark.parametrize('sample_func', \
    [ image_sample_points_nearest, image_sample_points_bilinear ])
def test_image_sample_points_float(sample_func):
    """ Check if sampling works with float images
    """
    image = np.arange(6).reshape((2, 3)).astype(float)
    points = np.array([[2.5, 1.5]])
    samples, _ = sample_func(image, points)
    assert np.all(samples == [5.0, ])
    assert samples.dtype == image.dtype



@pytest.mark.parametrize('sample_func', \
    [ image_sample_points_nearest, image_sample_points_bilinear ])
def test_image_sample_exact_points(sample_func):
    """ No matter what implementation of image sampling: if we use exact
    image indices, we expect no rounding, interpolation or whatever: just
    the exact value at that index!
    """
    # Generate test image
    num_rows = 3
    num_cols = 4
    rows = np.arange(num_rows)
    cols = np.arange(num_cols)
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    image = 100.0 * np.array(rows) + np.array(cols)
    # Re-use indices to generate points
    indices = np.zeros((image.size, 2))
    indices[:, 0] = rows.ravel()
    indices[:, 1] = cols.ravel()
    points = image_indices_to_points(indices)
    # Sample at points
    samples, on_chip_mask = sample_func(image, points)
    assert np.allclose(image.ravel(), samples)
    assert np.all(on_chip_mask)



def generate_2x2_test_image_and_points():
    """ Unique test case for all sampling methods
    Taken from https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    # Generate 2x2 image with different colors in each edge
    image = np.array((
        # Red        Green
        ((1, 0, 0), (0, 1, 0)),
        # Blue       Red
        ((0, 0, 1), (1, 0, 0)),
    ), dtype=np.float64)
    # Generate image points covering the image in high resolution
    margin = 0.5
    xmin = -margin
    xmax = image.shape[1] + margin
    xnum = 6
    ymin = -margin
    ymax = image.shape[0] + margin
    ynum = 6
    x = np.linspace(xmin, xmax, xnum)
    y = np.linspace(ymin, ymax, ynum)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    points = np.zeros((xnum * ynum, 2))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    return image, points



def test_image_sample_points_nearest_basic():
    image, points = generate_2x2_test_image_and_points()
    samples, on_chip_mask = image_sample_points_nearest(image, points)
    expected_samples = np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [1., 0., 0.],
        [1., 0., 0.],
    ])
    assert np.allclose(samples, expected_samples)
    expected_on_chip_mask = np.array([
        False, False, False, False, False, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False, False, False, False, False, False,
    ])
    assert np.all(on_chip_mask == expected_on_chip_mask)



def test_image_sample_points_bilinear_basic():
    image, points = generate_2x2_test_image_and_points()
    samples, on_chip_mask = image_sample_points_bilinear(image, points)
    expected_samples = np.array([
        [1.  , 0.  , 0.  ],
        [0.8 , 0.2 , 0.  ],
        [0.2 , 0.8 , 0.  ],
        [0.  , 1.  , 0.  ],
        [0.8 , 0.  , 0.2 ],
        [0.68, 0.16, 0.16],
        [0.32, 0.64, 0.04],
        [0.2 , 0.8 , 0.  ],
        [0.2 , 0.  , 0.8 ],
        [0.32, 0.04, 0.64],
        [0.68, 0.16, 0.16],
        [0.8 , 0.2 , 0.  ],
        [0.  , 0.  , 1.  ],
        [0.2 , 0.  , 0.8 ],
        [0.8 , 0.  , 0.2 ],
        [1.  , 0.  , 0.  ],
    ])
    assert np.allclose(samples, expected_samples)
    expected_on_chip_mask = np.array([
        False, False, False, False, False, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False,  True,  True,  True,  True, False,
        False, False, False, False, False, False,
    ])
    assert np.all(on_chip_mask == expected_on_chip_mask)



def test_image_sample_points_nearest_corner_cases():
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(6).reshape((2, 3)) # Red channel
    EPS = 1e-6
    points = np.array([
        [ 1.0-EPS, 1.5 ],
        #[ 1.0, 1.5 ],
        [ 1.0+EPS, 1.5 ],
        [ 1.5,     1.5 ],
        [ 2.0-EPS, 1.5 ],
        #[ 2.0, 1.5 ],
        [ 2.0+EPS, 1.5 ],
    ])
    samples, _ = image_sample_points_nearest(image, points)
    samples = samples[:, 0] # Red channel
    expected_samples = [ 3.0, 4.0, 4.0, 4.0, 5.0 ]
    assert np.all(np.isclose(samples, expected_samples))