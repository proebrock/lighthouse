import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import scipy.sparse

from trafolib.trafo3d import Trafo3d



def _sparse_dilate(a, shape):
    result = scipy.sparse.lil_matrix(( \
        a.shape[0] * shape[0],
        a.shape[1] * shape[1]), dtype=a.dtype)
    rows, cols = a.nonzero()
    for row, col in zip(rows, cols):
        result[shape[0]*row:shape[0]*(row+1), \
               shape[1]*col:shape[1]*(col+1)] = a[row, col]
    return result

def _generate_sparsity_points(mask):
    sparsity = scipy.sparse.lil_matrix((np.sum(mask), mask.shape[0]), dtype=int)
    r = 0
    for c in range(mask.shape[0]):
        h = np.sum(mask[c, :])
        sparsity[r:r+h, c] = 1
        r += h
    return _sparse_dilate(sparsity, (2, 3))

def _generate_sparsity_poses(mask):
    sparsity = scipy.sparse.lil_matrix((np.sum(mask), mask.shape[1]), dtype=int)
    r = 0
    for i in range(mask.shape[0]):
        h = np.sum(mask[i, :])
        rows = np.arange(r, r+h)
        cols = np.where(mask[i, :])[0]
        sparsity[rows, cols] = 1
        r += h
    return _sparse_dilate(sparsity, (2, 6))

def _generate_sparsity_points_and_poses(mask):
    sparsity_points = _generate_sparsity_points(mask)
    sparsity_poses = _generate_sparsity_poses(mask)
    return scipy.sparse.hstack((sparsity_points, sparsity_poses))



def _objfun_bundle_adjust_points(x, cams, p, mask):
    P = x.reshape((-1, 3))
    p_estim = np.zeros_like(p)
    for i, cam in enumerate(cams):
        p_estim[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distance
    residuals = p_estim - p
    return residuals[mask, :].ravel()



def bundle_adjust_points(cams, p, P_init=None, full=False):
    """ Calculate bundle adjustment solving for 3D points
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
    :param full: Provide just points (False) or additionally residuals and distance (True)
    :return: 3D scene points, shape (m, 3); residuals, shape (m, n); distances, shape (m, n)
    """
    # Check consistency
    assert p.ndim == 3
    num_points = p.shape[0]
    assert p.shape[1] == len(cams)
    num_views = p.shape[1]
    assert p.shape[2] == 2
    assert p.dtype == float

    # Check if enough observations available for all points
    mask = np.all(np.isfinite(p), axis=2)
    mask_enough_observations = np.sum(mask, axis=1) >= 2
    if not np.all(mask_enough_observations):
        raise ValueError('Provide enough observations in p to solve bundle adjustment')

    # Initial estimates for the points
    if P_init is None:
        P_init = np.zeros((num_points, 3))
        P_init[:, 2] = 500
    else:
        assert P_init.ndim == 2
        assert P_init.shape[0] == num_points
        assert P_init.shape[1] == 3
    x0 = P_init.ravel()

    # Setup sparse matrix with mapping of decision variables influncing residuals;
    # helps optimizer to efficiently calculate the Jacobian
    sparsity = _generate_sparsity_points(mask)

    # Run numerical optimization
    # TODO: constraint optimization with Z>=0 ?
    # TODO: stable optimization with a loss function!?
    result = least_squares(_objfun_bundle_adjust_points, x0,
        args=(cams, p, mask), jac_sparsity=sparsity)
    if not result.success:
        raise Exception('Numerical optimization failed.')

    if False:
        residuals_x0 = _objfun_bundle_adjust_points(x0, cams, p, mask)
        residuals = _objfun_bundle_adjust_points(result.x, cams, p, mask)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(residuals_x0, label='before opt')
        ax.plot(residuals, label='after opt')
        ax.set_ylabel('Error (pixels)')
        ax.grid()
        ax.legend()
        plt.show()

    # Extract resulting points
    P = result.x.reshape((-1, 3))

    if not full:
        return P

    # Extract residuals
    residuals = np.empty_like(p)
    residuals[:] = np.NaN
    residuals[mask] = _objfun_bundle_adjust_points( \
        result.x, cams, p, mask).reshape((-1, 2))
    # Calculate distance on chip in pixels: sqrt(dx**2+dy**2)
    residuals = np.sqrt(np.sum(np.square(residuals), axis=2))

    # Extract distances of camera rays to estimated 3D points
    distances = np.zeros((num_points, num_views))
    distances[:] = np.NaN
    for i, cam in enumerate(cams):
        rays = cam.get_rays(p[:, i, :])
        distances[:, i] = rays.to_points_distances(P)

    return P, residuals, distances



def _param_to_x_objfun_bundle_adjust_points_and_poses(P, poses):
    poses_coeff = []
    for pose in poses:
        p = np.concatenate((pose.get_translation(),
            pose.get_rotation_rodrigues()))
        poses_coeff.append(p)
    poses_coeff = np.asarray(poses_coeff)
    x = np.concatenate((P.flatten(), poses_coeff.flatten()))
    return x



def _x_to_param_objfun_bundle_adjust_points_and_poses(x, num_points):
    P = x[:3*num_points].reshape((num_points, 3))
    poses_coeff = x[3*num_points:].reshape((-1, 6))
    num_views = poses_coeff.shape[0]
    poses = []
    for i in range(num_views):
        T = Trafo3d(t=poses_coeff[i, :3], rodr=poses_coeff[i, 3:])
        poses.append(T)
    return P, poses



def _objfun_bundle_adjust_points_and_poses(x, cam, p, mask):
    P, poses = _x_to_param_objfun_bundle_adjust_points_and_poses(x, p.shape[0])
    p_estim = np.zeros_like(p)
    for i, pose in enumerate(poses):
        cam.set_pose(pose)
        p_estim[:, i, :] = cam.scene_to_chip(P)[:, 0:2] # Omit distance
    residuals = p_estim - p
    return residuals[mask, :].ravel()



def bundle_adjust_points_and_poses(cam, p, P_init=None, pose_init=None, full=False):
    """ Calculate bundle adjustment solving for 3D points and poses
    """
    # Check consistency
    assert p.ndim == 3
    num_points = p.shape[0]
    num_views = p.shape[1]
    assert p.shape[2] == 2
    assert p.dtype == float

    # Check if enough observations available for all points
    mask = np.all(np.isfinite(p), axis=2)
    mask_enough_observations = np.sum(mask, axis=1) >= 2
    if not np.all(mask_enough_observations):
        raise ValueError('Provide enough observations in p to solve bundle adjustment')

    # Initial estimates for the points
    if P_init is None:
        P_init = np.zeros((num_points, 3))
        P_init[:, 2] = 500
    else:
        assert P_init.ndim == 2
        assert P_init.shape[0] == num_points
        assert P_init.shape[1] == 3
    if pose_init is None:
        pose_init = num_views * [ Trafo3d() ]
    else:
        assert len(pose_init) == num_views
    x0 = _param_to_x_objfun_bundle_adjust_points_and_poses(P_init, pose_init)

    # Setup sparse matrix with mapping of decision variables influncing residuals;
    # helps optimizer to efficiently calculate the Jacobian
    sparsity = _generate_sparsity_points_and_poses(mask)

    # Run numerical optimization
    # TODO: constraint optimization with Z>=0 ?
    # TODO: stable optimization with a loss function!?
    result = least_squares(_objfun_bundle_adjust_points_and_poses, x0,
        args=(cam, p, mask), jac_sparsity=sparsity)
    if not result.success:
        raise Exception('Numerical optimization failed.')

    if False:
        # For testing the correct setting of the sparsity matrix:
        # Run optimization without 'jac_sparsity=sparsity',
        # with considerably low number of points and views and
        # some invisible points; then compare both sparsity matrices
        # in the console output
        with np.printoptions(threshold=100000, linewidth=100000):
            print()
            print(sparsity.toarray()) # Our calcuation of sparsity
            sparsity2 = (result.jac != 0).astype(int)
            print(sparsity2.toarray()) # Resulting sparsity from optimization
            assert np.all(sparsity.toarray() == sparsity2.toarray())

    if False:
        residuals_x0 = _objfun_bundle_adjust_points_and_poses(x0, cam, p, mask)
        residuals = _objfun_bundle_adjust_points_and_poses(result.x, cam, p, mask)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(residuals_x0, label='before opt')
        ax.plot(residuals, label='after opt')
        ax.set_ylabel('Error (pixels)')
        ax.grid()
        ax.legend()
        plt.show()

    # Extract resulting points
    P, poses = _x_to_param_objfun_bundle_adjust_points_and_poses(result.x, num_points)

    if not full:
        return P, poses