from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.pyplot as plt



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

        :param dim: Dimension (0 or 1)
        :return: Stack of lines
        """
        pass



    @abstractmethod
    def match(self, images):
        pass



    @staticmethod
    def _binarize_image(image, background_image, verbose=False):
        # Subtract images and handle underflow properly
        diff_image = image.astype(float) - background_image.astype(float)
        diff_image[diff_image < 0] = 0
        diff_image = diff_image.astype(np.uint8)
        # Thresholding
        _, bimage = cv2.threshold(diff_image, 128, 192, cv2.THRESH_OTSU)
        # Convert image to boolean
        bimage = bimage > 0
        # Visualization (if requested)
        if verbose:
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.imshow(image.T, cmap='gray')
            ax.set_axis_off()
            ax.set_title('before binarization')
            ax = fig.add_subplot(122)
            ax.imshow(bimage.T, cmap='gray')
            ax.set_axis_off()
            ax.set_title('after binarization')
            plt.show()
        return bimage



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



    def match(self, images):
        if images.ndim <= 1:
            raise ValueError('Provide proper images')
        if images.shape[0] != self._power:
            raise ValueError('Provide correct number of images')
        img = images.reshape((images.shape[0], -1))
        factors = np.power(2, np.arange(images.shape[0])[::-1])
        indices = np.zeros((img.shape[1], ), dtype=int)
        for i in range(img.shape[1]):
            indices[i] = int(np.sum((img[:, i] / 255) * factors))
        return indices.reshape(images.shape[1:])



class ImageMatcher:

    def __init__(self, line_matcher, shape):
        self._row_matcher = line_matcher(shape[0])
        self._col_matcher = line_matcher(shape[1])



    def generate_images(self):
        """ Generates a stack of images
        Those images can be displayed sequentially by a projector or a screen
        and later

        The image stack has a shape of (k, l, m) and contains k images
        of shape (l, m). Data type is uint8 (grayscale).

        :return: Stack of images
        """
        images = np.empty((2 + self._row_matcher.num_lines() + self._col_matcher.num_lines(),
            self._row_matcher.num_pixels(), self._col_matcher.num_pixels()), dtype=np.uint8)
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
