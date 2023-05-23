import pytest
import numpy as np
import matplotlib.pyplot as plt

from common.image_utils import image_show_multiple
from common.pixel_matcher import LineMatcherBinary, LineMatcherPhaseShift, ImageMatcher



@pytest.fixture(params=[LineMatcherBinary, LineMatcherPhaseShift])
def LineMatcherImplementation(request):
    return request.param



def test_line_matcher_roundtrip(LineMatcherImplementation):
    # Generate lines
    n = 400
    matcher = LineMatcherImplementation(n)
    lines = matcher.generate()
    # Check generated lines
    assert lines.ndim == 2
    assert lines.shape[0] == matcher.num_lines()
    assert lines.shape[1] == n
    assert lines.dtype == np.uint8
    # Match
    indices = matcher.match(lines)
    assert indices.ndim == 1
    assert indices.shape[0] == n
    assert indices.dtype == float
    # Generate expected indices
    expected_indices = np.arange(n)
    # Compare with matches
    diff = np.round(indices).astype(int) - expected_indices
    assert np.all(diff == 0)
    # Change shape of lines and see of output still correct
    images = lines.reshape((-1, 25, 16))
    indices = matcher.match(images)
    assert indices.ndim == 2
    assert np.all(indices.shape == (25, 16))
    diff = np.round(indices).astype(int) - expected_indices.reshape((25, 16))
    assert np.all(diff == 0)



def generate_image_roundtrip_indices(shape):
    expected_indices = np.zeros((shape[0], shape[1], 2), dtype=int)
    i0 = np.arange(shape[0])
    i1 = np.arange(shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    expected_indices[:, :, 0] = i0
    expected_indices[:, :, 1] = i1
    return expected_indices



def test_image_matcher_roundtrip(LineMatcherImplementation):
    # Generate images
    shape = (60, 80)
    matcher = ImageMatcher(LineMatcherImplementation, shape)
    images = matcher.generate()
    #image_show_multiple(images, single_window=True)
    #plt.show()
    # Check generated images
    assert images.ndim == 3
    assert images.shape[1] == shape[0]
    assert images.shape[2] == shape[1]
    assert images.dtype == np.uint8
    # Match
    indices = matcher.match(images)
    assert np.all(indices.shape == (shape[0], shape[1], 2))
    assert indices.dtype == float
    # Check results
    expected_indices = generate_image_roundtrip_indices(shape)
    diff = np.round(indices).astype(int) - expected_indices
    assert np.all(diff == 0)



def test_image_matcher_roundtrip_with_reduced_dynamic_range(LineMatcherImplementation):
    # Generate images
    shape = (60, 80)
    matcher = ImageMatcher(LineMatcherImplementation, shape)
    images = matcher.generate()
    # Reduce dynamic range
    images = 64 + images // 2
    # Match
    indices = matcher.match(images)
    # Check results
    expected_indices = generate_image_roundtrip_indices(shape)
    diff = np.round(indices).astype(int) - expected_indices
    assert np.all(diff == 0)



def test_image_matcher_roundtrip_with_unmatchable_pixels():
    # Generate images
    shape = (60, 80)
    matcher = ImageMatcher(LineMatcherPhaseShift, shape)
    images = matcher.generate()
    # Make some pixels unmatchable
    unmatchable_mask = np.zeros(shape, dtype=bool)
    images[0:2, 5, 8] = 0 # Black and white both zero
    unmatchable_mask[5, 8] = True
    images[2:, 10, 21] = 137 # Constant brightness, bad fit
    unmatchable_mask[10, 21] = True
    # Match
    indices = matcher.match(images)
    # Check results
    nan_mask = np.any(np.isnan(indices), axis=2) # any index NaN
    assert np.all(nan_mask == unmatchable_mask)
    expected_indices = generate_image_roundtrip_indices(shape)
    diff = np.round(indices[~unmatchable_mask]).astype(int) - \
        expected_indices[~unmatchable_mask]
    assert np.all(diff == 0)
