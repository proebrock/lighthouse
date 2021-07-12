# -*- coding: utf-8 -*-
""" Model for radial lens distortion
"""

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt



class LensDistortionModel:
    """ Class with model for radial lens distortion

    Uses the OpenCV lens model, see under "Detailed Description"
    https://docs.opencv.org/master/d9/d0c/group__calib3d.html

    (0,  1,  2,  3,   4,   5,  6,  7,   8,  9,  10  11)    <- Indices in self.coef
    (k1, k2, p1, p2[, k3[, k4, k5, k6[, s1, s2, s3, s4]]]) <- OpenCV names
    k1-k6 Radial distortion coefficients
    p1-p2 Tangential distortion coefficients
    s1-s4 Thin prism distortion coefficients
    """
    def __init__(self, coef):
        """ Constructor
        :param coef: Distortion coefficients
        """
        self.coef = np.zeros(12)
        c = np.asarray(coef)
        if c.size > 12:
            raise Exception('Provide proper distortion coefficients')
        self.coef[0:c.size] = c



    @staticmethod
    def __plot_points(p_dist, p_undist, title=None):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(p_dist[:,0], p_dist[:,1], 'xr', label='distorted')
        ax.plot(p_undist[:,0], p_undist[:,1], 'xg', label='undistorted')
        ax.grid()
        ax.legend()
        if title is None:
            ax.set_title(title)
        plt.show()



    def undistort(self, p):
        """ Undistort a set of points
        :param p: n points to unistort, shape (n, 2)
        :return: n undistorted points
        """
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Provide coordinates of shape (n, 2)')
        rsq = np.sum(np.square(p), axis=1)
        rd = (1.0 + self.coef[0] * rsq + self.coef[1] * rsq**2 + self.coef[4] * rsq**3) / \
            (1.0 + self.coef[5] * rsq + self.coef[6] * rsq**2 + self.coef[7] * rsq**3)
        result = np.empty_like(p)
        result[:,0] = p[:,0]*rd + 2*self.coef[2]*p[:,0]*p[:,1] + self.coef[3]*(rsq+2*p[:,0]*p[:,0]) + \
            self.coef[8]*rsq + self.coef[9]*rsq*rsq
        result[:,1] = p[:,1]*rd + self.coef[2]*(rsq+2*p[:,1]*p[:,1]) + 2*self.coef[3]*p[:,0]*p[:,1] + \
            self.coef[9]*rsq + self.coef[10]*rsq*rsq
        return result



    def __objfun(self, x, p):
        return (self.undistort(x.reshape((-1, 2))) - p).ravel()



    def distort(self, p):
        """ Distort a set of points
        :param p: n points to distort, shape (n, 2)
        :return: n distorted points
        """
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('Provide coordinates of shape (n, 2)')
        x0 = np.copy(p).ravel()
        result = least_squares(self.__objfun, x0, args=(p, ))
        if not result.success:
            raise Exception(f'Numerically solving undistort failed: {result}')
        p_dist = result.x.reshape((-1, 2))
        if False:
            residuals = self.__objfun(result.x, p).reshape((-1, 2))
            residuals = np.sum(np.square(residuals), axis=1) # per point residual
            rms = np.sqrt(np.mean(np.square(residuals)))
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(residuals)
            ax.grid()
            ax.set_title(f'Residuals; reprojection error RMS: {rms:.2f} pixels')
            plt.show()
            LensDistortionModel.__plot_points(p_dist, p, 'Result of distort()')
        return p_dist
