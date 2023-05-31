import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix



def _objfun_bundle_adjust(x, cams, p, mask):
    P = x.reshape((-1, 3))
    p_estim = np.zeros_like(p)
    for i, cam in enumerate(cams):
        p_estim[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distance
    residuals = p_estim - p
    return residuals[mask, :].ravel()



def bundle_adjust(cams, p, Pinit=None):
    """ Calculate bundle adjustment
    The user provides an array p of 2D points of shape (m, n, 2).
    The i-th line of p with i in [0..m-1] contains the projection of a single unknown
    3D scene point onto the chips of up to n cameras. So p[i, j, :] contains the
    projection of said 3D point onto the chip of the j-th camera with j in [0..n-1].
    In total the bundle adjustments reconstructs m 3D scene points.
    If a point was not observed in a camera, it is marked as (NaN, NaN). If a point
    was not at least seen by 2 cameras, it cannot be reconstructed.
    The residuals contain the on-chip reprojection error sqrt(du**2 + dv**2)
    in pixels, one for each point and camera. Residual is NaN if no observation
    by given camera or not enough points for reconstruction.
    :param cams: List of n cameras
    :param p: 2D on-chip points, shape (m, n, 2)
    :param Pinit: Initial guesses for 3D point positions
    :return: 3D scene points, shape (m, 3); residuals, shape (m, n)
    """
    assert p.ndim == 3
    assert p.shape[1] == len(cams)
    assert p.shape[2] == 2
    assert p.dtype == float

    # Reduce points to those that have been seen by two or more cameras
    mask = np.all(np.isfinite(p), axis=2)
    p_valid = np.sum(mask, axis=1) >= 2
    p_reduced = p[p_valid, :, :]
    mask_reduced = np.all(np.isfinite(p_reduced), axis=2)

    # Initial estimates for the points
    if Pinit is None:
        Pinit = np.zeros((p.shape[0], 3))
        Pinit[:, 2] = 500
    else:
        assert Pinit.ndim == 2
        assert Pinit.shape[0] == p.shape[0]
        assert Pinit.shape[1] == 3
    x0 = Pinit[p_valid, :].ravel()

    # Setup sparse matrix with mapping of decision variables influncing residuals;
    # helps optimizer to efficiently calculate the Jacobian
    num_residuals = 2 * np.sum(mask_reduced)
    num_decision_variables = 3 * np.sum(p_valid)
    sparsity = lil_matrix((num_residuals, num_decision_variables), dtype=int)
    r = 0
    c = 0
    for i in range(mask_reduced.shape[0]):
        h = 2 * np.sum(mask_reduced[i])
        w = 3
        sparsity[r:r+h, c:c+w] = 1
        r += h
        c += w

    # Run numerical optimization
    # TODO: constraint optimization with Z>=0 ?
    # TODO: stable optimization with a loss function!?
    result = least_squares(_objfun_bundle_adjust, x0,
        args=(cams, p_reduced, mask_reduced), jac_sparsity=sparsity)
    if not result.success:
        raise Exception('Numerical optimization failed.')

    # Extract resulting points
    P = np.empty((p.shape[0], 3))
    P[:] = np.NaN
    P[p_valid] = result.x.reshape((-1, 3))

    # Extract residuals
    residuals_reduced = np.empty_like(p_reduced)
    residuals_reduced[:] = np.NaN
    residuals_reduced[mask_reduced] = _objfun_bundle_adjust( \
        result.x, cams, p_reduced, mask_reduced).reshape((-1, 2))
    residuals = np.empty_like(p)
    residuals[:] = np.NaN
    residuals[p_valid, :, :] = residuals_reduced
    # Calculate distance on chip in pixels: sqrt(dx**2+dy**2)
    residuals = np.sqrt(np.sum(np.square(residuals), axis=2))
    return P, residuals

