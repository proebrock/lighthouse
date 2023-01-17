import pytest
import numpy as np
import matplotlib.pyplot as plt

from pixel_matcher import PixelMatcherBinary



@pytest.fixture(params=[PixelMatcherBinary])
def PixelMatcherImplementation(request):
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



def test_generate_images_properties(PixelMatcherImplementation):
    pm = PixelMatcherImplementation((256, 128))
    images = pm.generate_images()
    #display_images(images)
    assert images.ndim == 3
    assert images.shape[0] > 2
    assert images.shape[1] == 256
    assert images.shape[2] == 128
    assert images.dtype == np.uint8
