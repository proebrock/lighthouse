import pytest
import numpy as np
import matplotlib.pyplot as plt

from pixel_matcher import LineMatcherBinary, ImageMatcher



@pytest.fixture(params=[LineMatcherBinary])
def LineMatcherImplementation(request):
    return request.param



# For debug purposes
def display_images(images):
    # Calculate optimal number of subplots (rows/cols) to display n images
    a = np.sqrt(images.shape[0] / 6.0)
    shape = np.ceil(np.array((3 * a, 2 * a))).astype(int)
    if (shape[0] - 1) * shape[1] >= images.shape[0]:
        shape[0] -= 1
    # One subplot per image in image stack
    fig = plt.figure()
    fig.tight_layout()
    for i in range(images.shape[0]):
        ax = fig.add_subplot(shape[0], shape[1], i+1)
        ax.imshow(images[i].T, cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_title(f'{i}')
    plt.show()



def test_line_matcher_roundtrip(LineMatcherImplementation):
    # Generate images
    n = 400
    pm = LineMatcherImplementation(n)
    lines = pm.generate()
    assert lines.ndim == 2
    assert lines.shape[1] == n
    assert lines.dtype == np.uint8
    # Feed lines into matcher; expecting identity
    indices = pm.match(lines)
    assert np.all(indices.shape == (400, ))
    assert indices.dtype == int
    diff = indices - np.arange(n)
    assert np.all(diff == 0)
    # Change shape of lines and see of output still correct
    images = lines.reshape((-1, 25, 16))
    indices = pm.match(images)
    assert np.all(indices.shape == (25, 16))
    diff = indices - np.arange(n).reshape((25, 16))
    assert np.all(diff == 0)



def test_image_matcher_roundtrip(LineMatcherImplementation):
    # Generate
    shape = (80, 60)
    pm = ImageMatcher(LineMatcherImplementation, shape)
    images = pm.generate()
    #display_images(images)
    assert images.ndim == 3
    assert images.shape[1] == shape[0]
    assert images.shape[2] == shape[1]
    assert images.dtype == np.uint8
    # Match
    indices = pm.match(images)
    assert indices.dtype == int
    # Compare result against expected indices
    expected_indices = np.zeros_like(indices, dtype=int)
    i0 = np.arange(shape[0])
    i1 = np.arange(shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    expected_indices[:, :, 0] = i0
    expected_indices[:, :, 1] = i1
    diff = indices - expected_indices
    assert np.all(diff == 0)
