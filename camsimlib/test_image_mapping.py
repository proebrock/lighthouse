import pytest

import numpy as np
import matplotlib.pyplot as plt

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



@pytest.mark.parametrize('sample_func', [ image_sample_points_nearest, ])
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
    samples, on_chip_mask = image_sample_points_nearest(image, points)
    assert np.allclose(image.ravel(), samples)
    assert np.all(on_chip_mask)



def test_image_sample_points_nearest_rgb():
    """ Check if sampling works with RGB images
    """
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[:, :, 0] = np.arange(6).reshape((2, 3)) # R
    image[:, :, 1] = np.arange(6).reshape((2, 3)) # G
    image[:, :, 2] = np.arange(6).reshape((2, 3)) # B
    points = np.array([[2.5, 1.5]])
    values, _ = image_sample_points_nearest(image, points)
    assert np.all(values == [5.0, 5.0, 5.0])



def test_image_sample_points_nearest_float():
    """ Check if sampling works with float images
    """
    image = np.arange(6).reshape((2, 3)).astype(float)
    points = np.array([[2.5, 1.5]])
    samples, _ = image_sample_points_nearest(image, points)
    assert np.all(samples == [5.0, ])



def test_image_sample_points_nearest_manual_points():
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



@pytest.mark.skip(reason="under construction")
def test_image_sample_gaga():
    # Generate 3x3 image with different colors in each edge
    image = np.array((
        # Red        Green      Red
        ((1, 0, 0), (0, 1, 0), (1, 0, 0)),
        # Blue       Red        Magenta
        ((0, 0, 1), (1, 0, 0), (1, 0, 1)),
        # Red        Cyan       Red
        ((1, 0, 0), (0, 1, 1), (1, 0, 0)),
    ))
    # Generate image points covering the image in high resolution
    n = 101
    margin = 0.1
    xmin = -margin
    xmax = image.shape[1] + margin
    ymin = -margin
    ymax = image.shape[0] + margin
    x = np.linspace(xmin, xmax, n)
    y = np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    points = np.zeros((n * n, 2))
    points[:, 0] = xx.ravel()
    points[:, 1] = yy.ravel()
    # Sample image
    values, on_chip_mask = image_sample_points_nearest(image, points)
    value_image = np.zeros((n*n, 3))
    value_image[on_chip_mask, :] = values
    # Convert values back into RGB image
    value_image = 255 * value_image.reshape((n, n, 3))
    value_image = value_image.astype(np.uint8)
    # Plot resulting image
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(value_image, extent=[xmin, xmax, ymax, ymin])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position('top')
    plt.show()
