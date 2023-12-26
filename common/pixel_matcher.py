from abc import ABC, abstractmethod
import os
import sys
import glob
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2

from common.image_utils import image_rgb_to_gray, image_gray_to_rgb



class LineMatcher(ABC):
    """ Abstract base class (ABC) for any line matcher
    """

    def __init__(self, num_pixels=100):
        """ Constructor
        :param num_pixels: Number of pixels to match
        """
        if not isinstance(num_pixels, int):
            raise ValueError('Provide number of pixels as integer')
        self._num_pixels = num_pixels



    def num_pixels(self):
        """ Get number of pixels to match
        :return: Number of pixels
        """
        return self._num_pixels



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        param_dict['num_pixels'] = self._num_pixels



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        self._num_pixels = param_dict['num_pixels']



    @abstractmethod
    def num_time_steps(self):
        """ Get number of time steps necessary to match the number of pixels
        Dependent on implementation; override in derived class
        :return: Number of time steps
        """
        pass



    @abstractmethod
    def generate(self):
        """ Generates stack of lines
        Shape is (num_time_steps, num_pixels, 3), dtype np.uint8 (RGB)
        Dependent on implementation; override in derived class
        :return: Stack of lines
        """
        pass



    @abstractmethod
    def _match(self, images, image_blk, image_wht):
        """ Match stack of image lines
        internal implementation without consistency checks and
        only dealing with 1-dimensional image lines (plus another dimension for RGB)
        Dependent on implementation; override in derived class
        :param images: Line stack to match, shape (num_time_steps, n, 3), dtype np.uint8
        :param image_blk: all black image, shape (n, 3), dtype np.uint8
        :param image_wht: all white image, shape (n, 3), dtype np.uint8
        :return: Indices refering to originally generated stack of lines,
            shape (n, ), dtype float (sub-pixel accuracy possible),
            invalid matches represented by np.NaN
        """
        pass



    def match(self, images, image_blk=None, image_wht=None):
        """ Match stack of image lines or images
        :param images: Line stack to match, shape (num_time_steps, n, m, 3),
                       3 color channels RGB, dtype np.uint8
        :param image_blk: all black image, shape (n, m, 3), dtype np.uint8
        :param image_wht: all white image, shape (n, m, 3), dtype np.uint8
        :return: Indices refering to originally generated stack of lines,
            shape (n, m), dtype float (sub-pixel accuracy possible),
            invalid matches represented by np.NaN
        """
        if images.ndim <= 2:
            raise ValueError('Provide proper images')
        if images.shape[0] != self.num_time_steps():
            raise ValueError('Provide correct number of images')
        if images.shape[-1] != 3:
            raise ValueError('Provide color images')
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
        indices = self._match( \
            images.reshape((self.num_time_steps(), -1, 3)),
            image_blk.reshape(-1, 3),
            image_wht.reshape(-1, 3))
        return indices.reshape(images.shape[1:-1])



class LineMatcherBinary(LineMatcher):
    """ Line matcher implementation based on binary patterns
    """

    def __init__(self, num_pixels=100):
        super(LineMatcherBinary, self).__init__(num_pixels)



    def num_time_steps(self):
        # Determine a power of two that: 2**power <= _num_pixels
        power = 0
        while 2**power < self._num_pixels:
            power += 1
        return power



    def generate(self):
        num_time_steps = self.num_time_steps()
        lines = np.zeros((num_time_steps, self._num_pixels), dtype=np.uint8)
        values = np.arange(self._num_pixels)
        for i in range(num_time_steps):
            mask = 1 << (num_time_steps - i - 1)
            lines[i, (values & mask) > 0] = 255
        return image_gray_to_rgb(lines)



    @staticmethod
    def _binarize_images(images, background_image):
        # Subtract images and handle underflow properly
        diff_images = images.astype(float) - background_image.astype(float)
        diff_images[diff_images < 0] = 0
        diff_images = diff_images.astype(np.uint8)
        # Thresholding
        bimages = diff_images >= 127
        return bimages



    @staticmethod
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



    def _match(self, images, image_blk, image_wht):
        img = image_rgb_to_gray(images)
        img_blk = image_rgb_to_gray(image_blk)
        img_wht = image_rgb_to_gray(image_wht)
        binary_images = self._binarize_images(img, img_blk)
        factors = np.zeros_like(binary_images, dtype=int)
        factors = np.power(2, np.arange(img.shape[0])[::-1])[:, np.newaxis]
        indices = np.sum(binary_images * factors, axis=0).astype(float)
        roi = self._binarize_images(img_wht, img_blk)
        indices[~roi] = np.NaN
        return indices



def phase_shift_debug_view(display_image, image_stack):
    assert display_image.ndim == 2
    assert image_stack.ndim == 3
    assert display_image.shape[0] == image_stack.shape[1]
    assert display_image.shape[1] == image_stack.shape[2]
    # Main plot that reacts to mouse-over events
    display_fig, display_ax = plt.subplots()
    display_ax.imshow(display_image)
    # Debug plot showing values of single pixel in time
    stack_fig, stack_ax = plt.subplots()
    indices = np.arange(image_stack.shape[0])
    dots, = stack_ax.plot(indices, image_stack[:, 0, 0], 'ob')

    def display_mouse_move(event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        row = np.round(y).astype(int)
        col = np.round(x).astype(int)
        points = image_stack[:, row, col]
        dots.set_data(indices, points)
        stack_fig.canvas.draw_idle()

    def display_close_event(event):
        plt.close(stack_fig)

    display_fig.canvas.mpl_connect('motion_notify_event', display_mouse_move)
    display_fig.canvas.mpl_connect('close_event', display_close_event)
    plt.show(block=True)



class LineMatcherPhaseShift(LineMatcher):
    """ Line matcher implementation based on phase shifted sine patterns
    """

    def __init__(self, num_pixels=100, num_time_steps=21, num_phases=2):
        super(LineMatcherPhaseShift, self).__init__(num_pixels)
        self._num_time_steps = num_time_steps
        self._margin = 0.05
        self._num_phases = num_phases



    def num_time_steps(self):
        return self._num_time_steps



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        super(LineMatcherPhaseShift, self).dict_save(param_dict)
        param_dict['num_time_steps'] = self._num_time_steps
        param_dict['margin'] = self._margin
        param_dict['num_phases'] = self._num_phases



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        super(LineMatcherPhaseShift, self).dict_load(param_dict)
        self._num_time_steps = param_dict['num_time_steps']
        self._margin = param_dict['margin']
        self._num_phases = param_dict['num_phases']



    @staticmethod
    def _model(phases, angles):
        values =  np.tile(phases, (len(angles), 1))
        values += np.tile(angles, (len(phases), 1)).T
        return np.sin(values)



    def generate(self):
        # Phase shift angles used to encode pixel index;
        # margin parameter is used to avoid ambiguities around 0°/360°
        phases = np.linspace(self._margin, 2 * np.pi - self._margin,
            self.num_pixels())
        # Angles
        angles = np.linspace(0, self._num_phases * 2 * np.pi,
            self._num_time_steps + 1)[:-1]
        values = self._model(phases, angles)
        lines = (255.0 * (values + 1.0)) / 2.0
        lines = lines.astype(np.uint8)
        return image_gray_to_rgb(lines)



    @staticmethod
    def _objfun(phases, values, angles):
        mvalues = LineMatcherPhaseShift._model(phases, angles)
        residuals = mvalues - values
        return np.sum(np.square(residuals), axis=0)

    @staticmethod
    def _sine_phase_fit(values, angles):
        assert values.shape[0] == angles.size
        phases0 = np.zeros(values.shape[1])
        sparsity = lil_matrix((values.shape[1], values.shape[1]), dtype=int)
        for i in range(values.shape[1]):
            sparsity[i, i] = 1
        result = least_squares(LineMatcherPhaseShift._objfun, phases0,
            args=(values, angles), jac_sparsity=sparsity)
        if not result.success:
            raise Exception(f'Optimization failed: {result.message}')
        residuals = LineMatcherPhaseShift._objfun(result.x, values, angles)
        residuals_rms = np.sqrt(residuals / angles.size)
        return result.x, residuals_rms



    def _match(self, images, image_blk, image_wht):
        # Scale image pixels to a range [-1..1]
        img = image_rgb_to_gray(images)
        img = img.astype(float)
        # Determine min/max of each pixel
        imin = np.min(img, axis=0)
        imax = np.max(img, axis=0)
        # Value of white pix at least n values higher than black
        valid = imax > (imin + 10.0)
        # Use black and white images to scale range to [0..1]
        img = (img[:, valid] - imin[valid]) / \
                (imax[valid] - imin[valid])
        # Clip
        img = np.clip(img, 0.0, 1.0)
        # Scale range from [0..1] to [-1, 1]
        img = 2.0 * img - 1.0

        angles = np.linspace(0, self._num_phases * 2 * np.pi,
            self._num_time_steps + 1)[:-1]
        # Fit sine functions along axis 0
        phi, res = self._sine_phase_fit(img, angles)
        # Collect fit result: phases
        phases = np.zeros(images.shape[1])
        phases[:] = np.NaN
        phases[valid] = (phi + 2*np.pi) % (2*np.pi) # Wrap to [0..2*pi]
        # Collect fit result: residuals (RMS)
        residuals_rms = np.zeros(images.shape[1])
        residuals_rms[:] = np.NaN
        residuals_rms[valid] = res
        # Calculate indices from phases
        indices = np.zeros(images.shape[1])
        indices[:] = np.NaN
        indices[valid] = ((phases[valid] - self._margin) * (self._num_pixels - 1)) / \
            (2*np.pi - 2*self._margin)
        if False:
            shape = (480, 640) # We dont have that information here, so provide manually
            values = residuals_rms.reshape(shape)
            images_debug = img.reshape((-1, *shape))
            phase_shift_debug_view(values, images_debug)
            sadf
        return indices



class ImageMatcher:

    def __init__(self, shape=(100, 160), row_matcher=None, col_matcher=None):
        if row_matcher is None:
            self._row_matcher = LineMatcherPhaseShift(shape[0])
        else:
            self._row_matcher = row_matcher
        if col_matcher is None:
            self._col_matcher = LineMatcherPhaseShift(shape[1])
        else:
            self._col_matcher = col_matcher



    def num_time_steps(self):
        return 2 + self._row_matcher.num_time_steps() + self._col_matcher.num_time_steps()



    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        # Row matcher
        param_dict['row_matcher'] = {}
        param_dict['row_matcher']['module_name'] = __name__
        param_dict['row_matcher']['class_name'] = self._row_matcher.__class__.__name__
        self._row_matcher.dict_save(param_dict['row_matcher'])
        # Column matcher
        param_dict['col_matcher'] = {}
        param_dict['col_matcher']['module_name'] = __name__
        param_dict['col_matcher']['class_name'] = self._col_matcher.__class__.__name__
        self._col_matcher.dict_save(param_dict['col_matcher'])



    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        # Row matcher
        module_name = param_dict['row_matcher']['module_name']
        class_name = param_dict['row_matcher']['class_name']
        cls = getattr(sys.modules[module_name], class_name)
        self._row_matcher = cls()
        self._row_matcher.dict_load(param_dict['row_matcher'])
        # Column matcher
        module_name = param_dict['col_matcher']['module_name']
        class_name = param_dict['col_matcher']['class_name']
        cls = getattr(sys.modules[module_name], class_name)
        self._col_matcher = cls()
        self._col_matcher.dict_load(param_dict['col_matcher'])



    def json_save(self, filename):
        """ Save object to json file
        :param filename: Filename (and path)
        """
        param_dict = {}
        self.dict_save(param_dict)
        with open(filename, 'w') as f:
            json.dump(param_dict, f, indent=4, sort_keys=True)



    def json_load(self, filename):
        """ Load object from json file
        :param filename: Filename (and path)
        """
        with open(filename) as f:
            param_dict = json.load(f)
        self.dict_load(param_dict)



    def generate(self):
        """ Generates a stack of images
        Those images can be displayed sequentially by a projector or a screen
        and later

        The image stack has a shape of (k, l, m, 3) and contains k images
        of shape (l, m). Data type is uint8 (RGB).

        :return: Stack of images
        """
        images = np.empty((
            self.num_time_steps(),
            self._row_matcher.num_pixels(),
            self._col_matcher.num_pixels(),
            3), dtype=np.uint8)
        images[0, :, :, :] = 0   # All black
        images[1, :, :, :] = 255 # All white
        # Row images
        offs = 2
        lines = self._row_matcher.generate()
        for i in range(lines.shape[0]):
            images[offs+i, :, :, :] = lines[i, :, np.newaxis]
        # Column images
        offs += lines.shape[0]
        lines = self._col_matcher.generate()
        for i in range(lines.shape[0]):
            images[offs+i, :, :, :] = lines[i, np.newaxis, :] # Col images
        return images



    def match(self, images):
        if images.ndim != 4:
            raise ValueError('Provide proper images')
        if images.shape[0] != self.num_time_steps():
            raise ValueError('Provide correct number of images')
        if images.shape[-1] != 3:
            raise ValueError('Provide color images')
        if images.dtype != np.uint8:
            raise ValueError('Provide images of correct type')
        indices = -1 * np.ones((images.shape[1], images.shape[2], 2))
        n = 2 + self._row_matcher.num_time_steps()
        indices[:, :, 0] = self._row_matcher.match(images[2:n], images[0], images[1])
        indices[:, :, 1] = self._col_matcher.match(images[n:],  images[0], images[1])
        return indices



def display_and_snap(display_images, cam_index):
    # Configure cam
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise Exception('Unable to open camera device')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)
    ret, image = cap.read()
    if not ret:
        raise Exception('Unable to read from camera device')
    shape = image.shape
    # Configure display
    # cv2.namedWindow('window', cv2.WND_PROP_FULLSCREEN)
    cv2.namedWindow('window', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN)
    # Prepare result
    cam_images = np.zeros((display_images.shape[0],
        shape[0], shape[1], 3), dtype=np.uint8)
    start = False
    for i in range(display_images.shape[0]):
        cv2.imshow("window", display_images[i])
        if not start:
            # First time wait for key press
            start = cv2.waitKey(0)
            start = True
        c = cv2.waitKey(100)
        if c & 0xff == ord('q'):
            print('Aborted by user.')
            break
        # Read multiple images to get rid of old buffered images
        for _ in range(50):
            ret, image = cap.read()
        cam_images[i] = image
    cap.release()
    cv2.destroyAllWindows()
    return cam_images



if __name__ == '__main__':
    data_path = 'matcher'
    MODE = 0
    if MODE == 0:
        # Generate images and feed those directly back into the matcher
        shape = (60, 80)
        num_time_steps = 11
        num_phases = 2
        row_matcher = LineMatcherPhaseShift(shape[0],
            num_time_steps, num_phases)
        col_matcher = LineMatcherPhaseShift(shape[1],
            num_time_steps, num_phases)
        matcher = ImageMatcher(shape, row_matcher, col_matcher)
        images = matcher.generate()
        matcher.match(images)
    elif MODE == 1:
        # Generate images, show on screen, take images by camera, save images
        shape = (200, 320)
        num_time_steps = 11
        num_phases = 2
        row_matcher = LineMatcherPhaseShift(shape[0],
            num_time_steps, num_phases)
        col_matcher = LineMatcherPhaseShift(shape[1],
            num_time_steps, num_phases)
        matcher = ImageMatcher(shape, row_matcher, col_matcher)
        display_images = matcher.generate()
        images = display_and_snap(display_images, 0)
        for i in range(images.shape[0]):
            retval = cv2.imwrite(os.path.join(data_path,
                        f'cam_image{i:04}.png'), images[i])
            if not retval:
                raise Exception(f'Error writing image')
    elif MODE == 2:
        # Load images and start matching
        filenames = sorted(glob.glob(os.path.join(data_path, 'cam_image????.png')))
        if len(filenames) == 0:
            raise Exception('No files to read')
        images = []
        for filename in filenames:
            image = cv2.imread(filename, 0)
            if image is None:
                raise Exception(f'Error reading image {filename}')
            images.append(image)
        images = np.array(images)
        # Matching
        shape = (200, 320)
        num_time_steps = 11
        num_phases = 2
        row_matcher = LineMatcherPhaseShift(shape[0],
            num_time_steps, num_phases)
        col_matcher = LineMatcherPhaseShift(shape[1],
            num_time_steps, num_phases)
        matcher = ImageMatcher(shape, row_matcher, col_matcher)
        indices = matcher.match(images)
        # Visualize result
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(indices[:, :, 0])
        ax.set_title('Row index')
        ax = fig.add_subplot(122)
        ax.imshow(indices[:, :, 1])
        ax.set_title('Column index')
        plt.show()

