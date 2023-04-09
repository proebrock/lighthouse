from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import cv2



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
        if images.shape[0] != self.num_lines():
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
        n = self.num_lines()
        indices = self._match( \
            images.reshape((n, -1)),
            image_blk.ravel(),
            image_wht.ravel())
        return indices.reshape(images.shape[1:])



class LineMatcherBinary(LineMatcher):

    def __init__(self, num_pixels):
        super(LineMatcherBinary, self).__init__(num_pixels)
        # Determine two numbers that 2**_power <= _num_pixels
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
        factors = np.zeros_like(binary_images, dtype=int)
        factors = np.power(2, np.arange(images.shape[0])[::-1])[:, np.newaxis]
        indices = np.sum(binary_images * factors, axis=0).astype(float)
        roi = _binarize_images(image_wht, image_blk)
        indices[~roi] = np.NaN
        return indices



class LineMatcherPhaseShift(LineMatcher):

    def __init__(self, num_pixels):
        super(LineMatcherPhaseShift, self).__init__(num_pixels)
        self._num_lines = 5
        self._margin = 0.05
        self._phases = np.linspace(self._margin, 2*np.pi-self._margin, self._num_pixels)
        self._angles = np.linspace(0, 2*np.pi, self._num_lines + 1)[:-1]



    def num_lines(self):
        return self._num_lines



    @staticmethod
    def _model(phases, angles):
        values = np.tile(phases, (len(angles), 1))
        values += np.tile(angles, (len(phases), 1)).T
        return np.sin(values)



    def generate(self):
        values = self._model(self._phases, self._angles)
        lines = (255.0 * (values + 1.0)) / 2.0
        lines = lines.astype(np.uint8)
        return lines



    @staticmethod
    def _objfun(phases, values, angles):
        mvalues = LineMatcherPhaseShift._model(phases, angles)
        residuals = mvalues - values
        return np.sum(np.square(residuals), axis=0)

    @staticmethod
    def _sine_phase_fit(values, angles):
        phases0 = np.zeros(values.shape[1])
        sparsity = lil_matrix((values.shape[1], values.shape[1]), dtype=int)
        for i in range(values.shape[1]):
            sparsity[i, i] = 1
        result = least_squares(LineMatcherPhaseShift._objfun, phases0,
            args=(values, angles), jac_sparsity=sparsity)
        if not result.success:
            raise Exception(f'Optimization failed: {result.message}')
        residuals = LineMatcherPhaseShift._objfun(result.x, values, angles)
        return result.x, residuals



    def _match(self, images, image_blk, image_wht):
        print(images.shape)
        values = (images.astype(float) - image_blk.astype(float)) / \
            (image_wht.astype(float) - image_blk.astype(float))
        values = np.clip(values, 0.0, 1.0)
        values = 2.0 * values - 1.0 # Scale to range [-1, 1]
        phases, residuals = self._sine_phase_fit(values, self._angles)
        # Wrap to range of [0..2*pi]
        phases = (phases + 2*np.pi) % (2*np.pi)
        # Calculate indices from phases
        indices = ((phases - self._margin) * (self._num_pixels - 1)) / (2*np.pi - 2*self._margin)
        if False:
            fig = plt.figure()
            ax = fig.add_subplot(131)
            ax.plot(phases)
            ax.set_title('phases')
            ax = fig.add_subplot(132)
            ax.plot(residuals)
            ax.set_title('residuals')
            ax = fig.add_subplot(133)
            ax.plot(indices)
            ax.set_title('indices')
            plt.show()
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
        indices = -1 * np.ones((images.shape[1], images.shape[2], 2))
        n = 2 + self._row_matcher.num_lines()
        indices[:, :, 0] = self._row_matcher.match(images[2:n], images[0], images[1])
        indices[:, :, 1] = self._col_matcher.match(images[n:],  images[0], images[1])
        return indices
