from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.pyplot as plt



# TODO: This is not a good place to keep this; should be configurable by user
def _binarize_images(images, background_image):
    # Subtract images and handle underflow properly
    diff_images = images.astype(float) - background_image.astype(float)
    diff_images[diff_images < 0] = 0
    diff_images = diff_images.astype(np.uint8)
    # Thresholding
    bimages = diff_images >= 127
    return bimages

def _binarize_images2(images, background_image):
    # Subtract images and handle underflow properly
    diff_images = images.astype(float) - background_image.astype(float)
    diff_images[diff_images < 0] = 0
    diff_images = diff_images.astype(np.uint8)
    # Thresholding
    if diff_images.ndim == 1:
        _, bimages = cv2.threshold(diff_images.reshape((-1, 1)), 128, 192, cv2.THRESH_OTSU)
        bimages = bimages.ravel()
    elif diff_images.ndim == 2:
        _, bimages = cv2.threshold(diff_images, 128, 192, cv2.THRESH_OTSU)
    elif diff_images.ndim == 3:
        bimages = np.zeros_like(diff_images, dtype=np.uint8)
        for i in range(bimages.shape[0]):
            _, img = cv2.threshold(diff_images[i, :], 128, 192, cv2.THRESH_OTSU)
            bimages[i] = img
    else:
        raise NotImplementedError
    # Convert image to boolean
    bimages = bimages > 0
    return bimages



class LineMatcher(ABC):

    def __init__(self, num_pixels):
        if not isinstance(num_pixels, int):
            raise ValueError('Provide number of pixels as integer')
        self._num_pixels = num_pixels



    def num_pixels(self):
        return self._num_pixels



    @abstractmethod
    def num_lines(self):
        pass



    @abstractmethod
    def generate(self, dim):
        """ Generates stack of lines

        The line stack has a shape of (k, l) and contains k lines
        of length l. Data type is uint8 (grayscale).

        Must be implemented in derived class

        :return: Stack of lines
        """
        pass



    @abstractmethod
    def _match(self, images, image_blk, image_wht):
        pass



    def match(self, images, image_blk=None, image_wht=None):
        if images.ndim <= 1:
            raise ValueError('Provide proper images')
        if images.shape[0] != self._power:
            raise ValueError('Provide correct number of images')
        if images.dtype != np.uint8:
            raise ValueError('Provide images of correct type')
        if image_blk is None:
            image_blk = np.zeros_like(images[0], dtype=np.uint8)
        elif not np.all(images[0].shape == image_blk.shape):
            raise ValueError('Provide properly shaped black image')
        if image_wht is None:
            image_wht = 255 * np.ones_like(images[0], dtype=np.uint8)
        elif not np.all(images[0].shape == image_wht.shape):
            raise ValueError('Provide properly shaped white image')
        return self._match(images, image_blk, image_wht)



class LineMatcherBinary(LineMatcher):

    def __init__(self, num_pixel):
        super(LineMatcherBinary, self).__init__(num_pixel)
        # Determine two numbers that 2**_power <= _num_pixel
        self._power = 0
        while 2**self._power < self._num_pixels:
            self._power += 1



    def num_lines(self):
        return self._power



    def generate(self):
        lines = np.zeros((self._power, self._num_pixels), dtype=np.uint8)
        values = np.arange(self._num_pixels)
        for i in range(self._power):
            mask = 1 << (self._power - i - 1)
            lines[i, (values & mask) > 0] = 255
        return lines



    def _match(self, images, image_blk, image_wht):
        binary_images = _binarize_images(images, image_blk)
        binary_images = binary_images.reshape((images.shape[0], -1))
        factors = np.zeros_like(binary_images, dtype=int)
        factors = np.power(2, np.arange(images.shape[0])[::-1])[:, np.newaxis]
        indices = np.sum(binary_images * factors, axis=0).astype(int)
        indices = indices.reshape(images.shape[1:])
        roi = _binarize_images(image_wht, image_blk)
        indices[~roi] = -1
        return indices



class ImageMatcher:

    def __init__(self, line_matcher, shape):
        self._row_matcher = line_matcher(shape[0])
        self._col_matcher = line_matcher(shape[1])



    def num_images(self):
        return 2 + self._row_matcher.num_lines() + self._col_matcher.num_lines()



    def generate(self):
        """ Generates a stack of images
        Those images can be displayed sequentially by a projector or a screen
        and later

        The image stack has a shape of (k, l, m) and contains k images
        of shape (l, m). Data type is uint8 (grayscale).

        :return: Stack of images
        """
        images = np.empty((
            self.num_images(),
            self._row_matcher.num_pixels(),
            self._col_matcher.num_pixels()), dtype=np.uint8)
        images[0, :, :] = 0   # All black
        images[1, :, :] = 255 # All white
        # Row images
        offs = 2
        lines = self._row_matcher.generate()
        for i in range(lines.shape[0]):
            images[offs+i, :, :] = lines[i, :, np.newaxis]
        # Column images
        offs += lines.shape[0]
        lines = self._col_matcher.generate()
        for i in range(lines.shape[0]):
            images[offs+i, :, :] = lines[i, np.newaxis, :] # Col images
        return images



    def match(self, images):
        if images.ndim != 3:
            raise ValueError('Provide proper images')
        if images.shape[0] != self.num_images():
            raise ValueError('Provide correct number of images')
        if images.shape[1] != self._row_matcher.num_pixels():
            raise ValueError('Provide correct shape of images')
        if images.shape[2] != self._col_matcher.num_pixels():
            raise ValueError('Provide correct shape of images')
        if images.dtype != np.uint8:
            raise ValueError('Provide images of correct type')
        indices = -1 * np.ones((images.shape[1], images.shape[2], 2), dtype=int)
        n = 2 + self._row_matcher.num_lines()
        indices[:, :, 0] = self._row_matcher.match(images[2:n], images[0], images[1])
        indices[:, :, 1] = self._col_matcher.match(images[n:],  images[0], images[1])
        return indices
