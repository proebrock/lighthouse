import numpy as np
from scipy.optimize import least_squares



def _objfun_bundle_adjust(x, cams, points, mask):
    P = x.reshape((-1, 3))
    points_estim = np.zeros_like(points)
    for i, cam in enumerate(cams):
        points_estim[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distance
    residuals = points_estim - points
    return residuals[mask, :].ravel()



def bundle_adjust(cams, points):
    assert points.ndim == 3
    assert points.shape[1] == len(cams)
    assert points.shape[2] == 2

    # Reduce points to those that have been seen by two or more cameras
    mask = np.all(np.isfinite(points), axis=2)
    points_valid = np.sum(mask, axis=1) >= 2
    points_reduced = points[points_valid, :, :]
    mask_reduced = np.all(np.isfinite(points_reduced), axis=2)

    # Initial estimates for the points; TODO: provide initial estimates by param
    P_init = np.zeros((np.sum(points_valid), 3))
    P_init[:, 2] = 500
    x0 = P_init.ravel()

    # TODO: sparse jacobian
    result = least_squares(_objfun_bundle_adjust, x0,
        args=(cams, points_reduced, mask_reduced))
    if not result.success:
        raise Exception('Numerical optimization failed.')

    # Extract resulting points
    P = np.empty((points.shape[0], 3))
    P[:] = np.NaN
    P[points_valid] = result.x.reshape((-1, 3))

    # Extract residuals
    residuals_reduced = np.empty_like(points_reduced)
    residuals_reduced[:] = np.NaN
    residuals_reduced[mask_reduced] = _objfun_bundle_adjust( \
        result.x, cams, points_reduced, mask_reduced).reshape((-1, 2))
    residuals = np.empty_like(points)
    residuals[:] = np.NaN
    residuals[points_valid, :, :] = residuals_reduced
    # Calculate distance on chip in pixels: sqrt(dx**2+dy**2)
    residuals = np.sqrt(np.sum(np.square(residuals), axis=2))
    return P, residuals

