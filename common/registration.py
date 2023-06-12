import numpy as np

from trafolib.trafo3d import Trafo3d



def orthogonal_procrustes(p0, p1):
    """ Estimate orthogonal matrix R mapping p0 to p1
    Expects correspondences between both point clouds:
    Point p0[i,:] must correspond to p1[:,i] for all i in [0..num_points].
    See https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
    and https://en.wikipedia.org/wiki/Kabsch_algorithm
    :param p0: First point cloud, shape (num_points, dim)
    :param p1: Second point cloud, shape same as p0
    :return: Rotation matrix R
    """
    assert p0.shape == p1.shape
    assert p0.shape[1] == 3 # Just tested for 3D
    H = np.dot(p0.T, p1)
    # Singular value decomposition
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # Special reflection case
    if np.linalg.det(R) < 0:
        Vt[p0.shape[1]-1, :] *= -1
        R = np.dot(Vt.T, U.T)
    return R.T



def estimate_transform(p0, p1, estimate_scale=False):
    """ Estimate trafo between two point clouds
    Expects correspondences between both point clouds:
    Point p0[i,:] must correspond to p1[:,i] for all i in [0..num_points].
    :param p0: First point cloud, shape (num_points, dim)
    :param p1: Second point cloud, shape same as p0
    :return: Trafo3d object of estimated transformation from p0 to p1
    """
    assert p0.shape == p1.shape
    # Estimate translations and de-mean
    m0 = np.mean(p0, axis=0)
    m1 = np.mean(p1, axis=0)
    pp0 = p0 - m0
    pp1 = p1 - m1
    # Estimate rotation matrix
    R = orthogonal_procrustes(pp0, pp1)
    # Estimate scale
    if estimate_scale:
        s0 = np.linalg.norm(pp0)
        s1 = np.linalg.norm(pp1)
        scale = s0 / s1
        t = m0 - np.dot(R, scale * m1)
        return Trafo3d(t=t, mat=R), scale
    else:
        t = m0 - np.dot(R, m1)
        return Trafo3d(t=t, mat=R)

