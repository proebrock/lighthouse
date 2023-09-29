# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
from . trafo2d import Trafo2d



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


@pytest.fixture
def test_cases(random_generator):
    result = []
    for angle in np.deg2rad(np.linspace(-315.0, 315.0, 15)):
        t = random_generator.uniform(-10.0, 10.0, (2, ))
        c = np.cos(angle)
        s = np.sin(angle)
        mat = np.array([[c, -s], [s, c]])
        hm = np.identity(3)
        hm[0:2, 2] = t
        hm[0:2, 0:2] = mat
        result.append([t, mat, hm, angle])
    return result


def test_conversions(test_cases):
    for _, mat, hm, angle in test_cases:
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


def test_inversions(test_cases):
    for _, _, hm, _ in test_cases:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo.inverse()
        hm2 = trafo2.get_homogeneous_matrix()
        assert np.allclose(hm, np.linalg.inv(hm2))


def test_multiply_self(test_cases):
    for _, _, hm, _ in test_cases:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        trafo2 = trafo * trafo.inverse()
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(3))
        trafo2 = trafo.inverse() * trafo
        assert np.allclose(trafo2.get_homogeneous_matrix(), np.identity(3))


def test_multiply_transformations(test_cases):
    for i in range(len(test_cases) - 1):
        (_, _, hm1, _) = test_cases[i]
        (_, _, hm2, _) = test_cases[i+1]
        trafo1 = Trafo2d()
        trafo1.set_homogeneous_matrix(hm1)
        trafo2 = Trafo2d()
        trafo2.set_homogeneous_matrix(hm2)
        trafo = trafo1 * trafo2
        assert np.allclose(trafo.get_homogeneous_matrix(), np.dot(hm1, hm2))


def test_multiply_single_points(random_generator, test_cases):
    for _, _, hm, _ in test_cases:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        p = random_generator.uniform(-10.0, 10.0, (2, ))
        p2 = trafo * p
        ph = np.append(p, 1.0)
        p3 = np.dot(hm, ph)[0:2]
        assert np.allclose(p2, p3)


def test_multiply_multiple_points(random_generator, test_cases):
    for _, _, hm, _ in test_cases:
        trafo = Trafo2d()
        trafo.set_homogeneous_matrix(hm)
        n = 2 + random_generator.integers(10)
        p = random_generator.uniform(-10.0, 10.0, (n, 2))
        p2 = trafo * p
        for i in range(n):
            assert np.allclose(trafo * p[i, :], p2[i, :])


def test_average_translations():
    t1 = Trafo2d(t=(10, -20))
    t2 = Trafo2d(t=(20, 40))
    assert Trafo2d.average([t1]) == t1
    assert Trafo2d.average([t1], weights=[1.0]) == t1
    assert Trafo2d.average([t1, t2], weights=[1.0, 0.0]) == t1
    assert Trafo2d.average([t1, t2], weights=[0.0, 1.0]) == t2
    assert Trafo2d.average([t1, t2]) == \
        Trafo2d(t=(15, 10))
    assert Trafo2d.average([t1, t2, t1, t2]) == \
        Trafo2d(t=(15, 10))
    assert Trafo2d.average([t1, t2], weights=[1.0, 1.0]) == \
        Trafo2d(t=(15, 10))
    assert Trafo2d.average([t1, t2], weights=[0.9, 0.1]) == \
        Trafo2d(t=(11, -14))
    assert Trafo2d.average([t1, t1, t1, t1, t1, t1, t1, t1, t1, t2]) == \
        Trafo2d(t=(11, -14))


def test_average_rotations():
    t1 = Trafo2d(angle=np.deg2rad(-40))
    t2 = Trafo2d(angle=np.deg2rad(60))
    assert Trafo2d.average([t1]) == t1
    assert Trafo2d.average([t1], weights=[1.0]) == t1
    assert Trafo2d.average([t1, t2], weights=[1.0, 0.0]) == t1
    assert Trafo2d.average([t1, t2], weights=[0.0, 1.0]) == t2
    assert Trafo2d.average([t1, t2]) == \
        Trafo2d(angle=np.deg2rad(10))


def test_dict_save_load_roundtrip():
    T0 = Trafo2d(t=(1, -2), angle=np.deg2rad(34.0))
    param_dict = {}
    T0.dict_save(param_dict)
    T1 = Trafo2d()
    T1.dict_load(param_dict)
    assert T0 == T1


def test_transform_points_with_nan():
    # Test setup
    T = Trafo2d(t=(1.0, -2.0), angle=np.deg2rad(45))
    p = np.array((
        (1.0, 2.0),
        (np.NaN, 2.0),
        (1.0, np.NaN),
        (np.NaN, np.NaN),
    ))
    nan_mask = np.array((
        (False, False),
        (True, True),
        (True, True),
        (True, True),
    ))
    # Test multiplication with shape (2, )
    for _p, mask in zip(p, nan_mask):
        p2 = T * p
        assert np.all(np.isnan(p2) == nan_mask)
    # Test multiplication with shape (N, 2)
    p2 = T * p
    assert np.all(np.isnan(p2) == nan_mask)
