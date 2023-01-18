from abc import ABC, abstractmethod
import numpy as np
import cv2
import matplotlib.pyplot as plt



class PixelMatcher(ABC):

    def __init__(self, shape):
        self._shape = np.asarray(shape, dtype=int)
        if self._shape.size != 2:
            raise ValueError('Provide 2d image shape')



    @abstractmethod
    def generate_lines(self, dim):
        """ Generates stack of lines

        The line stack has a shape of (k, l) and contains k lines
        of length l. Data type is uint8 (grayscale).

        Must be implemented in derived class

        :param dim: Dimension (0 or 1)
        :return: Stack of lines
        """
        pass



    def generate_images(self):
        """ Generates a stack of images
        Those images can be displayed sequentially by a projector or a screen
        and later

        The image stack has a shape of (k, l, m) and contains k images
        of shape (l, m). Data type is uint8 (grayscale).

        :return: Stack of images
        """
        lines0 = self.generate_lines(0)
        lines1 = self.generate_lines(1)
        images = np.empty((2 + lines0.shape[0] + lines1.shape[0],
            self._shape[0], self._shape[1]), dtype=np.uint8)
        images[0, :, :] = 0   # All black
        images[1, :, :] = 255 # All white
        # Row images
        offs = 2
        for i in range(lines0.shape[0]):
            images[offs+i, :, :] = lines0[i, :, np.newaxis]
        # Column images
        offs += lines0.shape[0]
        for i in range(lines1.shape[0]):
            images[offs+i, :, :] = lines1[i, np.newaxis, :] # Col images
        return images



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



    def match(self, images):
        pass



class PixelMatcherBinary(PixelMatcher):

    def __init__(self, shape):
        super(PixelMatcherBinary, self).__init__(shape)
        # Determine two numbers that 2**n_i <= _shape[i]
        self._powers = np.array((0, 0), dtype=int)
        while 2**self._powers[0] < self._shape[0]:
            self._powers[0] += 1
        while 2**self._powers[1] < self._shape[1]:
            self._powers[1] += 1



    def generate_lines(self, dim):
        lines = np.zeros((self._powers[dim], self._shape[dim]), dtype=np.uint8)
        values = np.arange(self._shape[dim])
        for i in range(self._powers[dim]):
            mask = 1 << (self._powers[dim] - i - 1)
            lines[i, (values & mask) > 0] = 255
        return lines



    def match_images(self, images):
        if images.shape[0] !=
