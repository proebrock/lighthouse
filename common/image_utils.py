import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt



def image_load(filename):
    image = cv2.imread(filename)
    if image is None:
        raise Exception(f'Error reading image {filename}')
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise Exception('Unexpected image format read')



def image_load_multiple(filenames_or_pattern):
    if isinstance(filenames_or_pattern, str):
        # Pattern
        filenames = sorted(glob.glob(filenames_or_pattern))
        if len(filenames) == 0:
            raise Exception(f'No filenames found under {filenames_or_pattern}')
    else:
        # List of files
        filenames = filenames_or_pattern
    images = []
    shape = None
    for filename in filenames:
        image = image_load(filename)
        if shape is None:
            shape = image.shape
        else:
            if ~np.all(shape == image.shape):
                raise Exception('Not all images have the same shape')
        images.append(image)
    if len(images) == 0:
        raise Exception('No images loaded from {filenames_or_pattern}')
    return np.array(images)



def image_show(image, title=None):
    fig = plt.figure()
    fig.tight_layout()
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)



def image_show_multiple(images, titles=None, single_window=False):
    if single_window:
        # Calculate optimal number of subplots (rows/cols) to display n images
        a = np.sqrt(images.shape[0] / 6.0)
        shape = np.ceil(np.array((2 * a, 3 * a))).astype(int)
        if (shape[0] - 1) * shape[1] >= images.shape[0]:
            shape[0] -= 1
        # One subplot per image in image stack
        fig = plt.figure()
        fig.tight_layout()
        for i in range(images.shape[0]):
            ax = fig.add_subplot(shape[0], shape[1], i+1)
            ax.imshow(images[i])
            ax.set_axis_off()
            if titles is None:
                ax.set_title(f'Image {i}')
            else:
                ax.set_title(titles[i])
    else:
        for i in range(images.shape[0]):
            fig = plt.figure()
            fig.tight_layout()
            ax = fig.add_subplot(111)
            ax.imshow(images[i])
            ax.set_axis_off()
            if titles is None:
                ax.set_title(f'Image {i}')
            else:
                ax.set_title(titles[i])



def image_save(filename, image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    retval = cv2.imwrite(filename, img)
    if not retval:
        raise Exception(f'Error writing image {filename}')



def image_3float_to_rgb(image, nan_color=(0, 255, 255)):
    # RGB image, but each channel encoded by float [0..1]
    assert image.ndim == 3
    assert image.shape[-1] == 3
    assert image.dtype == float
    valid_mask = np.all(np.isfinite(image), axis=-1)
    img = np.zeros_like(image, dtype=np.uint8)
    img[~valid_mask, :] = np.asarray(nan_color)
    img[valid_mask, :] = (255.0 * np.clip(0.0, 1.0, image[valid_mask, :])).astype(np.uint8)
    return img



def image_float_to_rgb(image, cmap_name='viridis', min_max=None, nan_color=(0, 255, 255)):
    # Each pixel encoded by a single float
    assert image.ndim == 2
    assert image.dtype == float
    valid_mask = np.isfinite(image)
    img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    img[~valid_mask, :] = np.asarray(nan_color)
    if min_max is None:
        min_max = (np.min(image[valid_mask]), np.max(image[valid_mask]))
    image_norm = np.clip((image[valid_mask] - min_max[0]) / (min_max[1] - min_max[0]), 0, 1)
    image_norm = 1.0 - image_norm # Invert: the closer, the higher the value
    cm = plt.get_cmap(cmap_name)
    colors = cm(image_norm)[:, 0:3]
    img[valid_mask, :] = np.round(255 * colors).astype(np.uint8)
    return img
