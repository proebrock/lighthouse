# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
from . trafo3d import Trafo3d


np.random.seed(0) # Make sure tests are repeatable


def rpy_to_matrix(rpy):
    rotX = np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rpy[0]), -np.sin(rpy[0])],
        [0.0, np.sin(rpy[0]), np.cos(rpy[0])]
        ])
    rotY = np.array([
        [np.cos(rpy[1]), 0.0, np.sin(rpy[1])],
        [0.0, 1.0, 0.0],
        [-np.sin(rpy[1]), 0.0, np.cos(rpy[1])]
        ])
    rotZ = np.array([
        [np.cos(rpy[2]), -np.sin(rpy[2]), 0.0],
        [np.sin(rpy[2]), np.cos(rpy[2]), 0.0],
        [0.0, 0.0, 1.0]
        ])
    return np.dot(rotZ, np.dot(rotY, rotX))


def matrix_to_rpy(R):
    q = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    if not np.isclose(q, 0.0):
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], q)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], q)
        z = 0.0
    return np.array([x, y, z])


def rpy_equal(rpy1, rpy2):
    mat1 = rpy_to_matrix(rpy1)
    mat2 = rpy_to_matrix(rpy2)
    return np.allclose(mat1, mat2)


def angles_valid(angles):
    if isinstance(angles, (list, np.ndarray)):
        return (np.asarray(angles) >= -np.pi).all() and (np.asarray(angles) <= np.pi).all()
    return np.abs(angles) <= np.pi


def matrix_to_quaternion(mat):
    # symmetric matrix K
    K = np.array([
        [mat[0, 0]-mat[1, 1]-mat[2, 2], 0.0, 0.0, 0.0],
        [mat[0, 1]+mat[1, 0], mat[1, 1]-mat[0, 0]-mat[2, 2], 0.0, 0.0],
        [mat[0, 2]+mat[2, 0], mat[1, 2]+mat[2, 1], mat[2, 2]-mat[0, 0]-mat[1, 1], 0.0],
        [mat[2, 1]-mat[1, 2], mat[0, 2]-mat[2, 0], mat[1, 0]-mat[0, 1],
         mat[0, 0]+mat[1, 1]+mat[2, 2]]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def quaternions_equal(q1, q2):
    return np.allclose(q1, q2) or np.allclose(q1, -1.0 * q2)


def quaternion_to_rodrigues(q):
    q = q / np.linalg.norm(q)
    angle = 2 * np.arccos(q[0])
    s = np.sqrt(1 - q[0]*q[0])
    if np.isclose(s, 0.0):
        return q[1:] * angle
    return (q[1:] * angle) / s


def rodrigues_equal(rodr1, rodr2):
    return np.allclose(rodr1, rodr2) or np.allclose(rodr1, -1.0 * rodr2)


def test_constructor():
    trafo = Trafo3d()
    assert np.allclose(trafo.get_homogeneous_matrix(), np.identity(4))
    trafo = Trafo3d(t=[1, 2, 3])
    assert np.allclose(trafo.get_translation(), np.array([1, 2, 3]))
    assert np.allclose(trafo.get_rotation_matrix(), np.identity(3))
    trafo = Trafo3d(t=[1, 2, 3], rpy=np.deg2rad([10, -20, 30]))
    assert np.allclose(trafo.get_translation(), np.array([1, 2, 3]))
    assert rpy_equal(trafo.get_rotation_rpy(), np.deg2rad([10, -20, 30]))
    trafo = Trafo3d(rpy=np.deg2rad([10, -20, 30]))
    assert np.allclose(trafo.get_translation(), np.zeros(3))
    assert rpy_equal(trafo.get_rotation_rpy(), np.deg2rad([10, -20, 30]))
    with pytest.raises(ValueError):
        trafo = Trafo3d(gaga=[1, 2, 3])
    with pytest.raises(ValueError):
        trafo = Trafo3d(t=[1, 2, 3], gaga=[1, 2, 3])
    with pytest.raises(ValueError):
        trafo = Trafo3d([1, 2, 3])
    with pytest.raises(ValueError):
        trafo = Trafo3d(t=[1, 2, 3], hom=np.identity(4))
    with pytest.raises(ValueError):
        trafo = Trafo3d(rpy=[10, -20, 30], hom=np.identity(4))
    with pytest.raises(ValueError):
        trafo = Trafo3d(rpy=[10, -20, 30], q=[1, 0, 0, 0])
    with pytest.raises(ValueError):
        trafo = Trafo3d(list=[0, 0, 0, 1, 0, 0, 0], q=[1, 0, 0, 0])
    with pytest.raises(ValueError):
        trafo = Trafo3d(list=[0, 0, 0])


def test_to_list():
    # trafo -> list -> trafo
    t = [1.0, 2.0, 3.0]
    rpy = np.deg2rad([10.0, 20.0, 330.0])
    trafo1 = Trafo3d(t=t, rpy=rpy)
    l = trafo1.to_list()
    trafo2 = Trafo3d(list=l)
    assert trafo1 == trafo2
    # list -> trafo -> list
    trafo1 = Trafo3d(list=l)
    l2 = trafo1.to_list()
    assert np.allclose(l, l2)


def test_set_translation_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_translation('teststring')
    with pytest.raises(ValueError):
        trafo.set_translation(np.array([1, 2]))
    trafo.set_translation([1.0, 2.0, 3.0])
    trafo.set_translation(np.array([1.0, 2.0, 3.0]))


def test_set_rotation_matrix_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_rotation_matrix('teststring')
    with pytest.raises(ValueError):
        trafo.set_rotation_matrix(np.arange(9))
    with pytest.raises(ValueError):
        trafo.set_rotation_matrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        trafo.set_rotation_matrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    trafo.set_rotation_matrix(np.identity(3).tolist())
    trafo.set_rotation_matrix(np.identity(3))


def test_set_homogeneous_matrix_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_homogeneous_matrix('teststring')
    with pytest.raises(ValueError):
        trafo.set_homogeneous_matrix(np.arange(16))
    trafo.set_homogeneous_matrix(np.identity(4).tolist())
    trafo.set_homogeneous_matrix(np.identity(4))


def test_set_rotation_rpy_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_rotation_rpy('teststring')
    with pytest.raises(ValueError):
        trafo.set_rotation_rpy(np.array([1, 2]))
    trafo.set_rotation_rpy([10.0, 20.0, -30.0])
    trafo.set_rotation_rpy(np.array([10.0, 20.0, -30.0]))


def test_set_rotation_quaternion_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_rotation_quaternion('teststring')
    with pytest.raises(ValueError):
        trafo.set_rotation_quaternion([1, 2])
    with pytest.raises(ValueError):
        trafo.set_rotation_quaternion([2.0, 0.0, 0.0, 0.0])
    trafo.set_rotation_quaternion([1.0, 0.0, 0.0, 0.0])
    trafo.set_rotation_quaternion(np.array([1.0, 0.0, 0.0, 0.0]))


def test_set_rotation_rodrigues_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        trafo.set_rotation_rodrigues('teststring')
    with pytest.raises(ValueError):
        trafo.set_rotation_rodrigues([1, 2])
    trafo.set_rotation_rodrigues([1.0, 0.0, 0.0])
    trafo.set_rotation_rodrigues(np.array([1.0, 0.0, 0.0]))


def test_multiply_input_validate():
    trafo = Trafo3d()
    with pytest.raises(ValueError):
        _ = trafo * 'teststring'
    with pytest.raises(ValueError):
        _ = trafo * np.zeros(4)
    with pytest.raises(ValueError):
        _ = trafo * np.zeros((4, 5))
    with pytest.raises(ValueError):
        _ = trafo * np.zeros((3, 5))
    _ = trafo * trafo
    _ = trafo * np.zeros(3)
    _ = trafo * np.zeros((1, 3))
    _ = trafo * np.zeros((5, 3))
    _ = trafo * np.zeros((1, 3)).tolist()
    _ = trafo * np.zeros((5, 3)).tolist()


def test_comparison():
    t1 = Trafo3d()
    t2 = Trafo3d()
    assert t1 == t2
    assert not t1 != t2
    assert t1 != "hello"
    assert not t1 == "hello"


def test_wrap_angles():
    assert np.isclose(Trafo3d.wrap_angles(0.0), 0.0)
    assert np.isclose(Trafo3d.wrap_angles(1.5*np.pi), -0.5*np.pi)
    assert np.isclose(Trafo3d.wrap_angles(3.0*np.pi), np.pi)
    assert np.isclose(Trafo3d.wrap_angles(-3.0*np.pi), np.pi)
    a = np.array([0.0, 1.5, 3.0, -3.0]) * np.pi
    b = np.array([0.0, -0.5, 1.0, 1.0]) * np.pi
    assert np.allclose(Trafo3d.wrap_angles(a), b)


def generate_rpy_testcases():
    result = []
    # Important corner cases
    phis = np.linspace(-315.0, 315.0, 15)
    for x_roll in phis:
        for y_pitch in phis:
            for z_yaw in phis:
                result.append(Trafo3d.wrap_angles(np.deg2rad(np.array([x_roll, y_pitch, z_yaw]))))
    # Random values
    for _ in range(100):
        result.append(2.0 * np.pi * 2.0 * (np.random.rand(3) - 0.5))
    return result


def generate_testcases():
    result = []
    for rpy in generate_rpy_testcases():
        t = 10.0 * 2.0 * (np.random.rand(3) - 0.5)
        mat = rpy_to_matrix(rpy)
        hm = np.identity(4)
        hm[0:3, 3] = t
        hm[0:3, 0:3] = mat
        q = matrix_to_quaternion(mat)
        rodr = quaternion_to_rodrigues(q)
        result.append([t, mat, hm, rpy, q, rodr])
    return result


# Auto-generated test-cases
TestCases = generate_testcases()
# Add manually generated test-cases
TestCases.append([
    np.array([-10.0, 15.0, 5.0]),
    np.array([
        [0.2548870022, 0.1621711752, -0.9532749478],
        [-0.0449434555, -0.9827840478, -0.179208262],
        [-0.9659258263, 0.0885213269, -0.2432103468]
        ]),
    np.array([
        [0.2548870022, 0.1621711752, -0.9532749478, -10.0],
        [-0.0449434555, -0.9827840478, -0.179208262, 15.0],
        [-0.9659258263, 0.0885213269, -0.2432103468, 5.0],
        [0.0, 0.0, 0.0, 1.0]
        ]),
    np.deg2rad(np.array([-20.0, 105.0, -190.0])),
    np.array([0.0849891282, 0.7875406969, 0.037213226, -0.6092386024]),
    np.array([2.34860312, 0.11097725, -1.81687079])
    ])
TestCases.append([
    np.array([110.0, 115.0, -200.0]),
    np.array([
        [-0.3213938048, -0.3562022041, -0.8773972943],
        [-0.8830222216, -0.2219212496, 0.4135489272],
        [-0.3420201433, 0.9076733712, -0.2432103468]
        ]),
    np.array([
        [-0.3213938048, -0.3562022041, -0.8773972943, 110.0],
        [-0.8830222216, -0.2219212496, 0.4135489272, 115.0],
        [-0.3420201433, 0.9076733712, -0.2432103468, -200.0],
        [0.0, 0.0, 0.0, 1.0]
        ]),
    np.deg2rad(np.array([285.0, -200.0, 70.0])),
    np.array([0.2310165572, 0.534728387, -0.579370974, -0.5701106708]),
    np.array([1.47035792, -1.59311292, -1.5676496])
    ])


def test_conversions():
    for _, mat, hm, rpy, q, rodr in TestCases:
        # From Matrix
        trafo = Trafo3d()
        trafo.set_rotation_matrix(mat)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        rpy2 = trafo.get_rotation_rpy()
        assert angles_valid(rpy2)
        assert rpy_equal(rpy, rpy2)
        q2 = trafo.get_rotation_quaternion()
        assert quaternions_equal(q, q2)
        # From homogenous matrix
        trafo = Trafo3d()
        trafo.set_homogeneous_matrix(hm)
        hm2 = trafo.get_homogeneous_matrix()
        assert np.allclose(hm, hm2)
        # From RPY
        trafo = Trafo3d()
        trafo.set_rotation_rpy(rpy)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        rpy2 = trafo.get_rotation_rpy()
        assert angles_valid(rpy2)
        assert rpy_equal(rpy, rpy2)
        q2 = trafo.get_rotation_quaternion()
        assert quaternions_equal(q, q2)
        rodr2 = trafo.get_rotation_rodrigues()
        assert rodrigues_equal(rodr, rodr2)
        # From Quaternion
        trafo = Trafo3d()
        trafo.set_rotation_quaternion(q)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        rpy2 = trafo.get_rotation_rpy()
        assert angles_valid(rpy2)
        assert rpy_equal(rpy, rpy2)
        q2 = trafo.get_rotation_quaternion()
        assert quaternions_equal(q, q2)
        rodr2 = trafo.get_rotation_rodrigues()
        assert rodrigues_equal(rodr, rodr2)
        # From Rodrigues
        trafo = Trafo3d()
        trafo.set_rotation_rodrigues(rodr)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        rpy2 = trafo.get_rotation_rpy()
        assert angles_valid(rpy2)
        assert rpy_equal(rpy, rpy2)
        q2 = trafo.get_rotation_quaternion()
        assert quaternions_equal(q, q2)
        rodr2 = trafo.get_rotation_rodrigues()
        assert rodrigues_equal(rodr, rodr2)


def test_inversions():
    for _, _, hm, _, _, _ in TestCases:
        trafo = Trafo3d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo.inverse()
        hm2 = trafo2.get_homogeneous_matrix()
        assert np.allclose(hm, np.linalg.inv(hm2))


def test_multiply_self():
    for _, _, hm, _, _, _ in TestCases:
        trafo = Trafo3d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo * trafo.inverse()
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(4))
        trafo2 = trafo.inverse() * trafo
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(4))


def test_multiply_transformations():
    for i in range(len(TestCases) - 1):
        (_, _, hm1, _, _, _) = TestCases[i]
        (_, _, hm2, _, _, _) = TestCases[i+1]
        trafo1 = Trafo3d()
        trafo1.set_homogeneous_matrix(hm1)
        trafo2 = Trafo3d()
        trafo2.set_homogeneous_matrix(hm2)
        trafo = trafo1 * trafo2
        assert np.allclose(trafo.get_homogeneous_matrix(), np.dot(hm1, hm2))


def test_multiplysingle_points():
    for _, _, hm, _, _, _ in TestCases:
        trafo = Trafo3d()
        trafo.set_homogeneous_matrix(hm)
        p = 10.0 * 2.0 * (np.random.rand(3) - 0.5)
        p2 = trafo * p
        ph = np.append(p, 1.0)
        p3 = np.dot(hm, ph)[0:3]
        assert np.allclose(p2, p3)


def test_multiply_multiple_points():
    for _, _, hm, _, _, _ in TestCases:
        trafo = Trafo3d()
        trafo.set_homogeneous_matrix(hm)
        n = 2 + np.random.randint(10)
        p = 10.0 * 2.0 * (np.random.rand(n, 3) - 0.5)
        p2 = trafo * p
        for i in range(n):
            assert np.allclose(trafo * p[i, :], p2[i, :])


def test_interpolate():
    t1 = np.array([10, 20, 10])
    r1 = np.deg2rad([30, 0, 0])
    trafo1 = Trafo3d(t=t1, rpy=r1)
    t2 = np.array([20, 10, -20])
    r2 = np.deg2rad([60, 0, 0])
    trafo2 = Trafo3d(t=t2, rpy=r2)

    trafo3 = trafo1.interpolate(trafo2, weight=0.0)
    assert np.allclose(t1, trafo3.get_translation())
    assert np.allclose(r1, trafo3.get_rotation_rpy())

    trafo3 = trafo1.interpolate(trafo2, weight=0.5)
    assert np.allclose((t1 + t2) / 2.0, trafo3.get_translation())
    assert np.allclose((r1 + r2) / 2.0, trafo3.get_rotation_rpy())

    trafo3 = trafo1.interpolate(trafo2, weight=1.0)
    assert np.allclose(t2, trafo3.get_translation())
    assert np.allclose(r2, trafo3.get_rotation_rpy())


def test_interpolate_multiple():
    trafos = []
    result = Trafo3d.interpolate_multiple(trafos)
    assert result is None

    t1 = np.array([10, 20, 10])
    r1 = np.deg2rad([90, 0, 0])
    trafo1 = Trafo3d(t=t1, rpy=r1)
    trafos.append(trafo1)
    result = Trafo3d.interpolate_multiple(trafos)
    assert result == trafo1

    t2 = np.array([20, 10, -20])
    r2 = np.deg2rad([270, 0, 0])
    trafo2 = Trafo3d(t=t2, rpy=r2)
    trafos.append(trafo2)
    result = Trafo3d.interpolate_multiple(trafos)
    assert result == trafo1.interpolate(trafo2, weight=0.5)

    t3 = np.array([100, 110, -120])
    r3 = np.deg2rad([0, 90, 0])
    trafo3 = Trafo3d(t=t3, rpy=r3)
    trafos.append(trafo3)
    result = Trafo3d.interpolate_multiple(trafos)
    assert result == trafo1.interpolate(trafo2, weight=1/2.0).interpolate(trafo3, weight=1/3.0)


def test_distance():
    t1 = np.array([10, 20, 10])
    r1 = np.deg2rad([90, 0, 0])
    trafo1 = Trafo3d(t=t1, rpy=r1)
    dt, dr = trafo1.distance(trafo1)
    assert dt == 0.0
    assert dr == 0.0

    t2 = np.array([10, 0, 0])
    r2 = np.deg2rad([0, 0, 0])
    trafo2 = Trafo3d(t=t2, rpy=r2)
    dt, dr = trafo2.distance(trafo2.inverse())
    assert dt == 20.0
    assert dr == 0.0

    t3 = np.array([0, 0, 0])
    r3 = np.deg2rad([0, 5, 0])
    trafo3 = Trafo3d(t=t3, rpy=r3)
    dt, dr = trafo3.distance(trafo3.inverse())
    assert dt == 0.0
    assert np.isclose(np.rad2deg(dr), 10.0)


def test_average_and_errors():
    trafos = []
    average, errors = Trafo3d.average_and_errors(trafos)
    assert average is None
    assert errors is None

    t1 = np.array([10, 20, 10])
    r1 = np.deg2rad([90, 0, 0])
    trafo1 = Trafo3d(t=t1, rpy=r1)
    trafos.append(trafo1)
    average, errors = Trafo3d.average_and_errors(trafos)
    assert average == trafo1
    assert np.allclose(errors, np.zeros(2))

    trafos.append(trafo1)
    average, errors = Trafo3d.average_and_errors(trafos)
    assert average == trafo1
    assert np.allclose(errors, np.zeros((2, 2)))
