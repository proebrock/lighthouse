""" Model for lens distortion
"""

import copy
import numpy as np
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import datetime



class LensDistortionModel:
    """ Class with model for lens distortion

    Uses the OpenCV lens model, see under "Detailed Description"
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html


    Lens distortion coefficients according to OpenCV model:
    (0,  1,  2,  3,   4,   5,  6,  7,   8,  9,  10  11)    <- Indices in self._coef
    (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]) <- OpenCV names
    k1-k6 Radial distortion coefficients
    p1-p2 Tangential distortion coefficients
    s1-s4 Thin prism distortion coefficients

    Setting a coefficent to zeros disables the according component.
    """
    def __init__(self, coef=None):
        """ Constructor
        :param coef: Distortion coefficients
        """
        if coef is None:
            self._coef = np.zeros(12)
        else:
            self.set_coefficients(coef)
        self._create_report = False



    def __str__(self):
        """ String representation of lens distortion model
        :return: String representing lens distortion model
        """
        if np.allclose(self._coef, 0.0):
            return '[]'
        else:
            with np.printoptions(precision=5, suppress=True):
                return f'{self._coef}'



    def __copy__(self):
        """ Shallow copy
        :return: A shallow copy of self
        """
        return self.__class__(coef=self._coef)



    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(coef=copy.deepcopy(self._coef, memo))
        memo[id(self)] = result
        return result



    def set_coefficients(self, coef):
        """ Set distortion coefficients
        :param coef: Distortion coefficients
        """
        self._coef = np.zeros(12)
        c = np.asarray(coef).flatten()
        # Reduce coefficients to number of supported coefficients
        if c.size > 12:
            c = c[:12]
        self._coef[0:c.size] = c



    def get_coefficients(self):
        """ Get distortion coefficients
        :return: Distortion coefficients
        """
        return self._coef



    def dict_save(self, param_dict):
        """ Save distortion coefficients to dictionary
        :param params: Dictionary to store distortion coefficients in
        """
        param_dict['distortion'] = self._coef.tolist()



    def dict_load(self, param_dict):
        """ Load camera parameters from dictionary
        :param params: Dictionary with camera parameters
        """
        self.set_coefficients(np.array(param_dict['distortion']))



    def undistort(self, p):
        """ Undistort a set of points
        :param p: n points to unistort, shape (n, 2)
        :return: n undistorted points
        """
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Provide coordinates of shape (n, 2)')
        if np.allclose(self._coef, 0.0):
            return p.copy()
        rsq = np.sum(np.square(p), axis=1)
        rd = (1.0 + self._coef[0] * rsq + self._coef[1] * rsq**2 + self._coef[4] * rsq**3) / \
            (1.0 + self._coef[5] * rsq + self._coef[6] * rsq**2 + self._coef[7] * rsq**3)
        p_undist = np.empty_like(p)
        p_undist[:,0] = p[:,0]*rd + \
            2*self._coef[2]*p[:,0]*p[:,1] + \
            self._coef[3]*(rsq+2*p[:,0]**2) + \
            self._coef[8]*rsq + self._coef[9]*rsq**2
        p_undist[:,1] = p[:,1]*rd + \
            self._coef[2]*(rsq+2*p[:,1]**2) + \
            2*self._coef[3]*p[:,0]*p[:,1] + \
            self._coef[10]*rsq + self._coef[11]*rsq**2
        return p_undist



    def __objfun(self, x, p_undist):
        """ Objective function for distort()
        :param x: Current guess for distorted points as flat array
        :param p_undist: Undistorted points
        :return: X/Y distances of undistorted x and real p_undist (flat)
        """
        return (self.undistort(x.reshape((-1, 2))) - p_undist).ravel()



    def distort(self, p):
        """ Distort a set of points
        :param p: n points to distort, shape (n, 2)
        :return: n distorted points
        """
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Provide coordinates of shape (n, 2)')
        if np.allclose(self._coef, 0.0):
            return p.copy()
        x0 = np.copy(p).ravel()
        sparsity = lil_matrix((p.size, p.size), dtype=int)
        for i in range(0, p.size, 2):
            sparsity[i, i] = 1
            sparsity[i+1, i] = 1
            sparsity[i, i+1] = 1
            sparsity[i+1, i+1] = 1
        result = least_squares(self.__objfun, x0, args=(p, ), jac_sparsity=sparsity)
        if not result.success:
            raise Exception(f'Numerically solving undistort failed: {result}')
        p_dist = result.x.reshape((-1, 2))

        if self._create_report:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]

            fig = plt.figure()
            ax = fig.add_subplot(111)
            residuals = self.__objfun(result.x, p).reshape((-1, 2))
            residuals = np.sum(np.square(residuals), axis=1) # per point residual
            rms = np.sqrt(np.mean(np.square(residuals)))
            ax.plot(residuals)
            ax.grid()
            ax.set_title(f'Residuals of distort(); reprojection error RMS: {rms:.2f} pixels')
            plt.close(fig)
            fig.savefig(timestamp + '_residuals.png', dpi=600, bbox_inches='tight')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            LensDistortionModel._plot_distortion_points(ax, p, p_dist, 'Distortion points')
            plt.close(fig)
            fig.savefig(timestamp + '_points.png', dpi=600, bbox_inches='tight')

            fig = plt.figure()
            ax = fig.add_subplot(111)
            LensDistortionModel._plot_distortion_field(ax, p, p_dist, 'Distortion field')
            plt.close(fig)
            fig.savefig(timestamp + '_field.png', dpi=600, bbox_inches='tight')

        return p_dist



    @staticmethod
    def _plot_distortion_points(ax, p_undist, p_dist, title=None):
        ax.plot(p_undist[:,0], p_undist[:,1], '+g', label='undistorted', alpha=0.5)
        ax.plot(p_dist[:,0], p_dist[:,1], '+r', label='distorted', alpha=0.5)
        ax.grid()
        ax.legend()
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)



    @staticmethod
    def _plot_distortion_field(ax, p_undist, p_dist, title=None, autoscale=True):
        delta = p_dist - p_undist
        if autoscale:
            ax.quiver(p_undist[:,0], p_undist[:,1], delta[:,0], delta[:,1])
        else:
            ax.quiver(p_undist[:,0], p_undist[:,1], delta[:,0], delta[:,1],
                angles='xy', scale_units='xy', scale=1)
        ax.set_aspect('equal')
        if title is not None:
            ax.set_title(title)

