import numpy as np
from scipy.optimize import least_squares



def _objfun_bundle_adjust(x, cams, points, points_cams_valid):
    P = x.reshape((-1, 3))
    points_estim = np.zeros_like(points)
    for i, cam in enumerate(cams):
        points_estim[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distance
    residuals = points_estim - points
    return residuals[points_cams_valid, :].ravel()



def bundle_adjust(cams, points):
    assert points.ndim == 3
    assert points.shape[1] == len(cams)
    assert points.shape[2] == 2

    # Reduce cameras to those that have seen at least a single point
    points_cams_valid = np.all(np.isfinite(points), axis=2)
    cams_valid = np.sum(points_cams_valid, axis=0) > 0
    cams_reduced = []
    for i, cam in enumerate(cams):
        if cams_valid[i]:
            cams_reduced.append(cam)
    # Reduce points to those that have been seen by two or more cameras
    points_valid = np.sum(points_cams_valid, axis=1) >= 2
    points_reduced = points[points_valid, :, :]
    points_reduced = points_reduced[:, cams_valid, :]

    P_init = np.zeros((np.sum(points_valid), 3))
    P_init[:, 2] = 500
    x0 = P_init.ravel()
    mask = np.all(np.isfinite(points_reduced), axis=2)

    # TODO: sparse jacobian
    result = least_squares(_objfun_bundle_adjust, x0,
        args=(cams_reduced, points_reduced, mask))
    if not result.success:
        raise Exception('Numerical optimization failed.')

    # Extract resulting points
    P = np.empty((points.shape[0], 3))
    P[:] = np.NaN
    P[points_valid] = result.x.reshape((-1, 3))

    # Extract residuals
    residuals_reduced = np.zeros_like(points_reduced)
    residuals_reduced[mask, :] = _objfun_bundle_adjust( \
        result.x, cams_reduced, points_reduced, mask).reshape((-1, 2))
    # TODO: how to summarize/normalize by number of cams

    return P

