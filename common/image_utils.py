import cv2
import glob



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



def image_save(filename, image):
    retval = cv2.imwrite(filename, image)
    if not retval:
        raise Exception(f'Error writing image {filename}')



def image_load_multiple(filenames_or_pattern):
    if isinstance(filenames_or_pattern, str):
        filenames = sorted(glob.glob(filenames_or_pattern))
    else:
        filenames = filenames_or_pattern
    return [ image_load(filename) for filename in filenames]
