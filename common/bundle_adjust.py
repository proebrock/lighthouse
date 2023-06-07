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



def bundle_adjust_points(cams, p, Pinit=None, full=False):
    """ Calculate bundle adjustment
    This bundle adjustment solver expects a number of cameras with fixed and
    known intrinsics and extrinsics and solves for a number of 3D points based
    on observations.
    The user provides an array p of 2D points of shape (m, n, 2).
    The i-th line of p with i in [0..m-1] contains the projection of a single unknown
    3D scene point onto the chips of up to n cameras. So p[i, j, :] contains the
    projection of said 3D i-th point onto the chip of the j-th camera with j in [0..n-1].
    In total the bundle adjustments reconstructs m 3D scene points.
    If a point was not observed in a camera, the user may mark it as (NaN, NaN).
    If any point was not seen by 2 or more cameras, it cannot be reconstructed and
    an exception is thrown.
    The residuals contain the on-chip reprojection error sqrt(du**2 + dv**2)
    in pixels, one for each point and camera. Residual is NaN if no observation
    by given camera.
    The distances are the distances of the camera rays to the finally reconstructed
    3D points.
    :param cams: List of n cameras
    :param p: 2D on-chip points, shape (m, n, 2)
    :param Pinit: Initial guesses for 3D point positions
    :return: 3D scene points, shape (m, 3); residuals, shape (m, n); distances, shape (m, n)
    """
    assert p.ndim == 3
    assert p.shape[1] == len(cams)
    assert p.shape[2] == 2
    assert p.dtype == float

    # Check if enough observations available for all points
    mask = np.all(np.isfinite(p), axis=2)
    mask_enough_observations = np.sum(mask, axis=1) >= 2
    if not np.all(mask_enough_observations):
        raise ValueError('Provide enough observations in p to solve bundle adjustment')

    # Initial estimates for the points
    if Pinit is None:
        Pinit = np.zeros((p.shape[0], 3))
        Pinit[:, 2] = 500
    else:
        assert Pinit.ndim == 2
        assert Pinit.shape[0] == p.shape[0]
        assert Pinit.shape[1] == 3
    x0 = Pinit.ravel()

    # Setup sparse matrix with mapping of decision variables influncing residuals;
    # helps optimizer to efficiently calculate the Jacobian
    num_residuals = 2 * np.sum(mask)
    num_decision_variables = 3 * p.shape[0]
    sparsity = lil_matrix((num_residuals, num_decision_variables), dtype=int)
    r = 0
    c = 0
    for i in range(p.shape[0]):
        h = 2 * np.sum(mask[i])
        w = 3
        sparsity[r:r+h, c:c+w] = 1
        r += h
        c += w
    #print(sparsity.toarray())
    #
    # Example for 4 cameras, 2 points with this visibility
    #     (True,  False, True,  False), # Point 2 visibile by 2 cameras
    #     (True,  True,  True,  False), # Point 3 visibile by 3 cameras
    #
    # Sparsity matrix:
    # [[1 1 1 0 0 0]
    #  [1 1 1 0 0 0]
    #  [1 1 1 0 0 0]
    #  [1 1 1 0 0 0]
    #  [0 0 0 1 1 1]
    #  [0 0 0 1 1 1]
    #  [0 0 0 1 1 1]
    #  [0 0 0 1 1 1]
    #  [0 0 0 1 1 1]
    #  [0 0 0 1 1 1]]
    #
    # For the first point (first 3 cols) we have 4 residuals: x/y for first
    # and third point.
    # For the second point (last 3 cols) we have 6 residuals: x/y for first
    # second and third point.

    # Run numerical optimization
    # TODO: constraint optimization with Z>=0 ?
    # TODO: stable optimization with a loss function!?
    result = least_squares(_objfun_bundle_adjust, x0,
        args=(cams, p, mask), jac_sparsity=sparsity)
    if not result.success:
        raise Exception('Numerical optimization failed.')

    # Extract resulting points
    P = result.x.reshape((-1, 3))

    if not full:
        return P

    # Extract residuals
    residuals = np.empty_like(p)
    residuals[:] = np.NaN
    residuals[mask] = _objfun_bundle_adjust( \
        result.x, cams, p, mask).reshape((-1, 2))
    # Calculate distance on chip in pixels: sqrt(dx**2+dy**2)
    residuals = np.sqrt(np.sum(np.square(residuals), axis=2))

    # Extract distances of camera rays to estimated 3D points
    distances = np.zeros((p.shape[0], p.shape[1]))
    distances[:] = np.NaN
    for i, cam in enumerate(cams):
        rays = cam.get_rays(p[:, i, :])
        distances[:, i] = rays.to_points_distances(P)

    return P, residuals, distances



def bundle_adjust_points_poses(cam, p):
    pass
