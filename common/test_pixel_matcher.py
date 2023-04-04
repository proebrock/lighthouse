import pytest
import numpy as np
import matplotlib.pyplot as plt

from pixel_matcher import LineMatcherBinary, ImageMatcher



@pytest.fixture(params=[LineMatcherBinary])
def LineMatcherImplementation(request):
    return request.param



# For debug purposes
def display_images(images):
    pass # Moved to common.image_utils



def generate_noisy_image(binary_image, blk_low, blk_high, wht_low, wht_high):
    assert binary_image.dtype == bool
    result = np.zeros_like(binary_image, dtype=float)
    result[binary_image] = np.random.uniform(wht_low, wht_high,
        result.shape)[binary_image]
    result[~binary_image] = np.random.uniform(blk_low, blk_high,
        result.shape)[~binary_image]
    result = np.clip(result, 0.0, 255.0).astype(np.uint8)
    return result



def test_line_matcher_roundtrip(LineMatcherImplementation):
    # Generate lines
    n = 400
    pm = LineMatcherImplementation(n)
    lines = pm.generate()
    # Check generated lines
    assert lines.ndim == 2
    assert lines.shape[1] == n
    assert lines.dtype == np.uint8
    # Modify lines
    lines = generate_noisy_image(lines > 0, 0, 50, 200, 250)
    # Match
    indices = pm.match(lines)
    assert np.all(indices.shape == (n, ))
    assert indices.dtype == int
    # Generate expected indices
    expected_indices = np.arange(n)
    # Compare with matches
    diff = indices - expected_indices
    assert np.all(diff == 0)
    # Change shape of lines and see of output still correct
    images = lines.reshape((-1, 25, 16))
    indices = pm.match(images)
    assert np.all(indices.shape == (25, 16))
    diff = indices - expected_indices.reshape((25, 16))
    assert np.all(diff == 0)



def test_image_matcher_roundtrip(LineMatcherImplementation):
    # Generate images
    shape = (800, 600)
    pm = ImageMatcher(LineMatcherImplementation, shape)
    images = pm.generate()
#    display_images(images)
    # Check generated images
    assert images.ndim == 3
    assert images.shape[1] == shape[0]
    assert images.shape[2] == shape[1]
    assert images.dtype == np.uint8
    # Modify images
    images = generate_noisy_image(images > 0, 0, 50, 200, 250)
    images[:, :, 400:420] = 142 # Set to constant to make matching impossible
    #display_images(images)
    # Match
    indices = pm.match(images)
    assert np.all(indices.shape == (shape[0], shape[1], 2))
    assert indices.dtype == int
    # Generate expected indices
    expected_indices = np.zeros_like(indices, dtype=int)
    i0 = np.arange(shape[0])
    i1 = np.arange(shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    expected_indices[:, :, 0] = i0
    expected_indices[:, :, 1] = i1
    expected_indices[:, 400:420, :] = -1 # No matching possible
    # Compare with matches
    diff = indices - expected_indices
    assert np.all(diff == 0)
