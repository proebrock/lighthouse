# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
from . trafo2d import Trafo2d


np.random.seed(0) # Make sure tests are repeatable


def test_constructor():
    trafo = Trafo2d()
    assert np.allclose(trafo.get_homogeneous_matrix(), np.identity(3))
    trafo = Trafo2d(t=[1, 2])
    assert np.allclose(trafo.get_translation(), np.array([1, 2]))
    assert np.allclose(trafo.get_rotation_matrix(), np.identity(2))
    trafo = Trafo2d(t=[1, 2], angle=np.deg2rad(67))
    assert np.allclose(trafo.get_translation(), np.array([1, 2]))
    assert np.isclose(trafo.get_rotation_angle(), np.deg2rad(67))
    trafo = Trafo2d(angle=np.deg2rad(-48))
    assert np.allclose(trafo.get_translation(), np.zeros(2))
    assert np.isclose(trafo.get_rotation_angle(), np.deg2rad(-48))
    with pytest.raises(ValueError):
        trafo = Trafo2d(gaga=[1, 2])
    with pytest.raises(ValueError):
        trafo = Trafo2d(t=[1, 2], gaga=[1, 2])
    with pytest.raises(ValueError):
        trafo = Trafo2d([1, 2])
    with pytest.raises(ValueError):
        trafo = Trafo2d(t=[1, 2], hom=np.identity(3))
    with pytest.raises(ValueError):
        trafo = Trafo2d(angle=0, hom=np.identity(3))
    with pytest.raises(ValueError):
        trafo = Trafo2d(mat=np.identity(2), hom=np.identity(3))
    with pytest.raises(ValueError):
        trafo = Trafo2d(angle=0, mat=np.identity(2))


def generate_testcases():
    result = []
    for angle in np.deg2rad(np.linspace(-315.0, 315.0, 15)):
        t = 10.0 * 2.0 * (np.random.rand(2) - 0.5)
        c = np.cos(angle)
        s = np.sin(angle)
        mat = np.array([[c, -s], [s, c]])
        hm = np.identity(3)
        hm[0:2, 2] = t
        hm[0:2, 0:2] = mat
        result.append([t, mat, hm, angle])
    return result

TESTCASES = generate_testcases()


def test_conversions():
    for _, mat, hm, angle in TESTCASES:
        # From Matrix
        trafo = Trafo2d()
        trafo.set_rotation_matrix(mat)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        angle2 = trafo.get_rotation_angle()
        assert np.isclose(Trafo2d.wrap_angle(angle),
                          Trafo2d.wrap_angle(angle2))
        # From homogenous matrix
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        hm2 = trafo.get_homogeneous_matrix()
        assert np.allclose(hm, hm2)
        # From angle
        trafo = Trafo2d()
        trafo.set_rotation_angle(angle)
        mat2 = trafo.get_rotation_matrix()
        assert np.allclose(mat, mat2)
        angle2 = trafo.get_rotation_angle()
        assert np.isclose(Trafo2d.wrap_angle(angle),
                          Trafo2d.wrap_angle(angle2))


def test_inversions():
    for _, _, hm, _ in TESTCASES:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo.inverse()
        hm2 = trafo2.get_homogeneous_matrix()
        assert np.allclose(hm, np.linalg.inv(hm2))


def test_multiply_self():
    for _, _, hm, _ in TESTCASES:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo * trafo.inverse()
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(3))
        trafo2 = trafo.inverse() * trafo
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(3))


def test_multiply_transformations():
    for i in range(len(TESTCASES) - 1):
        (_, _, hm1, _) = TESTCASES[i]
        (_, _, hm2, _) = TESTCASES[i+1]
        trafo1 = Trafo2d()
        trafo1.set_homogeneous_matrix(hm1)
        trafo2 = Trafo2d()
        trafo2.set_homogeneous_matrix(hm2)
        trafo = trafo1 * trafo2
        assert np.allclose(trafo.get_homogeneous_matrix(), np.dot(hm1, hm2))


def test_multiply_single_points():
    for _, _, hm, _ in TESTCASES:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        p = 10.0 * 2.0 * (np.random.rand(2) - 0.5)
        p2 = trafo * p
        ph = np.append(p, 1.0)
        p3 = np.dot(hm, ph)[0:2]
        assert np.allclose(p2, p3)


def test_multiply_multiple_points():
    for _, _, hm, _ in TESTCASES:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        n = 2 + np.random.randint(10)
        p = 10.0 * 2.0 * (np.random.rand(n, 2) - 0.5)
        p2 = trafo * p
        for i in range(n):
            assert np.allclose(trafo * p[i, :], p2[i, :])
