# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
from . Trafo2d import Trafo2d


np.random.seed(0) # Make sure tests are repeatable


def test_Constructor():
	trafo = Trafo2d()
	assert np.allclose(trafo.GetHomogeneousMatrix(), np.identity(3))
	trafo = Trafo2d(t=[1,2])
	assert np.allclose(trafo.GetTranslation(), np.array([1,2]))
	assert np.allclose(trafo.GetRotationMatrix(), np.identity(2))
	trafo = Trafo2d(t=[1,2], angle=np.deg2rad(67))
	assert np.allclose(trafo.GetTranslation(), np.array([1,2]))
	assert np.isclose(trafo.GetRotationAngle(), np.deg2rad(67))
	trafo = Trafo2d(angle=np.deg2rad(-48))
	assert np.allclose(trafo.GetTranslation(), np.zeros(2))
	assert np.isclose(trafo.GetRotationAngle(), np.deg2rad(-48))
	with pytest.raises(ValueError):
		trafo = Trafo2d(gaga=[1,2])
	with pytest.raises(ValueError):
		trafo = Trafo2d(t=[1,2], gaga=[1,2])
	with pytest.raises(ValueError):
		trafo = Trafo2d([1,2])
	with pytest.raises(ValueError):
		trafo = Trafo2d(t=[1,2], hom=np.identity(3))
	with pytest.raises(ValueError):
		trafo = Trafo2d(angle=0, hom=np.identity(3))
	with pytest.raises(ValueError):
		trafo = Trafo2d(mat=np.identity(2), hom=np.identity(3))
	with pytest.raises(ValueError):
		trafo = Trafo2d(angle=0, mat=np.identity(2))


def GenerateTestCases():
	result = []
	for angle in np.deg2rad(np.linspace(-315.0, 315.0, 15)):
		t = 10.0 * 2.0 * (np.random.rand(2) - 0.5)
		c = np.cos(angle)
		s = np.sin(angle)
		mat = np.array([ [ c, -s ], [ s, c ] ])
		hm = np.identity(3)
		hm[0:2,2] = t
		hm[0:2,0:2] = mat
		result.append([ t, mat, hm, angle])
	return result

TestCases = GenerateTestCases()


def test_Conversions():
	for t, mat, hm, angle in TestCases:
		# From Matrix
		trafo = Trafo2d()
		trafo.SetRotationMatrix(mat)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		angle2 = trafo.GetRotationAngle()
		assert np.isclose(Trafo2d.WrapAngle(angle),
			Trafo2d.WrapAngle(angle2))
		# From homogenous matrix
		trafo = Trafo2d()
		trafo.SetHomogeneousMatrix(hm)
		hm2 = trafo.GetHomogeneousMatrix()
		assert np.allclose(hm, hm2)
		# From angle
		trafo = Trafo2d()
		trafo.SetRotationAngle(angle)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		angle2 = trafo.GetRotationAngle()
		assert np.isclose(Trafo2d.WrapAngle(angle),
			Trafo2d.WrapAngle(angle2))


def test_Inversions():
	for t, mat, hm, angle in TestCases:
		trafo = Trafo2d()
		trafo.SetHomogeneousMatrix(hm)
		trafo2 = trafo.Inverse()
		hm2 = trafo2.GetHomogeneousMatrix()
		assert np.allclose(hm, np.linalg.inv(hm2))


def test_MultiplySelf():
	for t, mat, hm, angle in TestCases:
		trafo = Trafo2d()
		trafo.SetHomogeneousMatrix(hm)
		trafo2 = trafo * trafo.Inverse()
		assert np.allclose(trafo2.GetHomogeneousMatrix(), np.identity(3))
		trafo2 = trafo.Inverse() * trafo
		assert np.allclose(trafo2.GetHomogeneousMatrix(), np.identity(3))


def test_MultiplyTransformations():
	for i in range(len(TestCases) - 1):
		(t1, mat1, hm1, angle1) = TestCases[i]
		(t2, mat2, hm2, angle2) = TestCases[i+1]
		trafo1 = Trafo2d()
		trafo1.SetHomogeneousMatrix(hm1)
		trafo2 = Trafo2d()
		trafo2.SetHomogeneousMatrix(hm2)
		trafo = trafo1 * trafo2
		assert np.allclose(trafo.GetHomogeneousMatrix(), np.dot(hm1, hm2))


def test_MultiplySinglePoints():
	for t, mat, hm, angle in TestCases:
		trafo = Trafo2d()
		trafo.SetHomogeneousMatrix(hm)
		p = 10.0 * 2.0 * (np.random.rand(2) - 0.5)
		p2 = trafo * p
		ph = np.append(p, 1.0)
		p3 = np.dot(hm, ph)[0:2]
		assert np.allclose(p2, p3)


def test_MultiplyMultiplePoints():
	for t, mat, hm, angle in TestCases:
		trafo = Trafo2d()
		trafo.SetHomogeneousMatrix(hm)
		n = 2 + np.random.randint(10)
		p = 10.0 * 2.0 * (np.random.rand(n,2) - 0.5)
		p2 = trafo * p
		for i in range(n):
			assert np.allclose(trafo * p[i,:], p2[i,:])

