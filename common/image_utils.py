import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt



def image_load(filename, black_white=False):
    image = cv2.imread(filename)
    if image is None:
        raise Exception(f'Error reading image {filename}')
    if image.ndim == 2:
        if black_white:
            return image
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        if black_white:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        raise Exception('Unexpected image format read')



def image_load_multiple(filenames_or_pattern, black_white=False):
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
        image = image_load(filename, black_white)
        if shape is None:
            shape = image.shape
        else:
            if ~np.all(shape == image.shape):
                raise Exception('Not all images have the same shape.')
        images.append(image)
    return np.array(images)



def image_show_multiple(images):
    # Calculate optimal number of subplots (rows/cols) to display n images
    a = np.sqrt(images.shape[0] / 6.0)
    shape = np.ceil(np.array((2 * a, 3 * a))).astype(int)
    if (shape[0] - 1) * shape[1] >= images.shape[0]:
        shape[0] -= 1
    print(shape)
    # One subplot per image in image stack
    fig = plt.figure()
    fig.tight_layout()
    for i in range(images.shape[0]):
        ax = fig.add_subplot(shape[0], shape[1], i+1)
        ax.imshow(images[i], cmap='gray', vmin=0, vmax=255)
        ax.set_axis_off()
        ax.set_title(f'Image {i}')
    plt.show()



def image_save(filename, image):
    retval = cv2.imwrite(filename, image)
    if not retval:
        raise Exception(f'Error writing image {filename}')
