# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
from . Trafo3d import Trafo3d


np.random.seed(0) # Make sure tests are repeatable


def RPY2Matrix(rpy):
	rotX = np.array([
		[ 1.0, 0.0, 0.0 ],
		[ 0.0, np.cos(rpy[0]), -np.sin(rpy[0]) ],
		[ 0.0, np.sin(rpy[0]), np.cos(rpy[0]) ]
		])
	rotY = np.array([
		[ np.cos(rpy[1]), 0.0, np.sin(rpy[1]) ],
		[ 0.0, 1.0, 0.0 ],
		[ -np.sin(rpy[1]), 0.0, np.cos(rpy[1]) ]
		])
	rotZ = np.array([
		[ np.cos(rpy[2]), -np.sin(rpy[2]), 0.0 ],
		[ np.sin(rpy[2]), np.cos(rpy[2]), 0.0 ],
		[ 0.0, 0.0, 1.0 ]
		])
	return np.dot(rotZ, np.dot(rotY, rotX))


def Matrix2RPY(R):
	q = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
	if not np.isclose(q, 0.0):
		x = np.arctan2(R[2,1], R[2,2])
		y = np.arctan2(-R[2,0], q)
		z = np.arctan2(R[1,0], R[0,0])
	else:
		x = np.arctan2(-R[1,2], R[1,1])
		y = np.arctan2(-R[2,0], q)
		z = 0.0
	return np.array([x, y, z])


def RPYsEqual(rpy1, rpy2):
	mat1 = RPY2Matrix(rpy1)
	mat2 = RPY2Matrix(rpy2)
	return np.allclose(mat1, mat2)


def AnglesValid(angles):
	if isinstance(angles, np.ndarray) or isinstance(angles, list):
		return (np.asarray(angles) >= -np.pi).all() and (np.asarray(angles) <= np.pi).all()
	else:
		return (angles >= -np.pi) and (angles <= np.pi)


def Mat2Quaternion(mat):
	# symmetric matrix K
	K = np.array([
		[ mat[0,0]-mat[1,1]-mat[2,2], 0.0,                        0.0,                        0.0                        ],
		[ mat[0,1]+mat[1,0],          mat[1,1]-mat[0,0]-mat[2,2], 0.0,                        0.0                        ],
		[ mat[0,2]+mat[2,0],          mat[1,2]+mat[2,1],          mat[2,2]-mat[0,0]-mat[1,1], 0.0                        ],
		[ mat[2,1]-mat[1,2],          mat[0,2]-mat[2,0],          mat[1,0]-mat[0,1],          mat[0,0]+mat[1,1]+mat[2,2] ]])
	K /= 3.0
	# quaternion is eigenvector of K that corresponds to largest eigenvalue
	w, V = np.linalg.eigh(K)
	q = V[[3, 0, 1, 2], np.argmax(w)]
	if q[0] < 0.0:
		np.negative(q, q)
	return q


def QuaternionsEqual(q1, q2):
	return np.allclose(q1, q2) or np.allclose(q1, -1.0 * q2)


def Quaternion2Rodrigues(q):
	q = q / np.linalg.norm(q)
	angle = 2 * np.arccos(q[0])
	s = np.sqrt(1 - q[0]*q[0])
	if np.isclose(s, 0.0):
		return (q[1:] * angle)
	else:
		return (q[1:] * angle) / s


def RodriguesEqual(rodr1, rodr2):
	return np.allclose(rodr1, rodr2) or np.allclose(rodr1, -1.0 * rodr2)


def test_Constructor():
	trafo = Trafo3d()
	assert np.allclose(trafo.GetHomogeneousMatrix(), np.identity(4))
	trafo = Trafo3d(t=[1,2,3])
	assert np.allclose(trafo.GetTranslation(), np.array([1,2,3]))
	assert np.allclose(trafo.GetRotationMatrix(), np.identity(3))
	trafo = Trafo3d(t=[1,2,3], rpy=np.deg2rad([10,-20,30]))
	assert np.allclose(trafo.GetTranslation(), np.array([1,2,3]))
	assert RPYsEqual(trafo.GetRotationRPY(), np.deg2rad([10,-20,30]))
	trafo = Trafo3d(rpy=np.deg2rad([10,-20,30]))
	assert np.allclose(trafo.GetTranslation(), np.zeros(3))
	assert RPYsEqual(trafo.GetRotationRPY(), np.deg2rad([10,-20,30]))
	with pytest.raises(ValueError):
		trafo = Trafo3d(gaga=[1,2,3])
	with pytest.raises(ValueError):
		trafo = Trafo3d(t=[1,2,3], gaga=[1,2,3])
	with pytest.raises(ValueError):
		trafo = Trafo3d([1,2,3])
	with pytest.raises(ValueError):
		trafo = Trafo3d(t=[1,2,3], hom=np.identity(4))
	with pytest.raises(ValueError):
		trafo = Trafo3d(rpy=[10,-20,30], hom=np.identity(4))
	with pytest.raises(ValueError):
		trafo = Trafo3d(rpy=[10,-20,30], q=[1,0,0,0])
	with pytest.raises(ValueError):
		trafo = Trafo3d(list=[0,0,0, 1,0,0,0], q=[1,0,0,0])
	with pytest.raises(ValueError):
		trafo = Trafo3d(list=[0,0,0])


def test_ToList():
	# trafo -> list -> trafo
	t = [1.0, 2.0, 3.0]
	rpy = np.deg2rad([10.0, 20.0, 330.0])
	trafo1 = Trafo3d(t=t, rpy=rpy)
	l = trafo1.ToList()
	trafo2 = Trafo3d(list=l)
	assert trafo1 == trafo2
	# list -> trafo -> list
	trafo1 = Trafo3d(list=l)
	l2 = trafo1.ToList()
	assert np.allclose(l, l2)


def test_SetTranslationInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetTranslation('teststring')
	with pytest.raises(ValueError):
		trafo.SetTranslation(np.array([1, 2]))
	trafo.SetTranslation([1.0, 2.0, 3.0])
	trafo.SetTranslation(np.array([1.0, 2.0, 3.0]))


def test_SetRotationMatrixInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetRotationMatrix('teststring')
	with pytest.raises(ValueError):
		trafo.SetRotationMatrix(np.arange(9))
	with pytest.raises(ValueError):
		trafo.SetRotationMatrix([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
	with pytest.raises(ValueError):
		trafo.SetRotationMatrix([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
	trafo.SetRotationMatrix(np.identity(3).tolist())
	trafo.SetRotationMatrix(np.identity(3))


def test_SetHomogeneousMatrixInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetHomogeneousMatrix('teststring')
	with pytest.raises(ValueError):
		trafo.SetHomogeneousMatrix(np.arange(16))
	trafo.SetHomogeneousMatrix(np.identity(4).tolist())
	trafo.SetHomogeneousMatrix(np.identity(4))


def test_SetRotationRPYInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetRotationRPY('teststring')
	with pytest.raises(ValueError):
		trafo.SetRotationRPY(np.array([1, 2]))
	trafo.SetRotationRPY([10.0, 20.0, -30.0])
	trafo.SetRotationRPY(np.array([10.0, 20.0, -30.0]))


def test_SetRotationQuaternionInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetRotationQuaternion('teststring')
	with pytest.raises(ValueError):
		trafo.SetRotationQuaternion([1, 2])
	with pytest.raises(ValueError):
		trafo.SetRotationQuaternion([2.0, 0.0, 0.0, 0.0])
	trafo.SetRotationQuaternion([1.0, 0.0, 0.0, 0.0])
	trafo.SetRotationQuaternion(np.array([1.0, 0.0, 0.0, 0.0]))


def test_SetRotationRodriguesInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo.SetRotationRodrigues('teststring')
	with pytest.raises(ValueError):
		trafo.SetRotationRodrigues([1, 2])
	trafo.SetRotationRodrigues([1.0, 0.0, 0.0])
	trafo.SetRotationRodrigues(np.array([1.0, 0.0, 0.0]))


def test_MultiplyInputValidate():
	trafo = Trafo3d()
	with pytest.raises(ValueError):
		trafo * 'teststring'
	with pytest.raises(ValueError):
		trafo * np.zeros(4)
	with pytest.raises(ValueError):
		trafo * np.zeros((4,5))
	with pytest.raises(ValueError):
		trafo * np.zeros((3,5))
	trafo * trafo
	trafo * np.zeros(3)
	trafo * np.zeros((1,3))
	trafo * np.zeros((5,3))
	trafo * np.zeros((1,3)).tolist()
	trafo * np.zeros((5,3)).tolist()


def test_Comparison():
	trafo = Trafo3d()
	assert trafo == trafo
	assert not trafo != trafo
	assert trafo != "hello"
	assert not trafo == "hello"


def test_WrapAngles():
	assert np.isclose(Trafo3d.WrapAngles(0.0), 0.0)
	assert np.isclose(Trafo3d.WrapAngles(1.5*np.pi), -0.5*np.pi)
	assert np.isclose(Trafo3d.WrapAngles(3.0*np.pi), np.pi)
	assert np.isclose(Trafo3d.WrapAngles(-3.0*np.pi), np.pi)
	a = np.array([0.0, 1.5, 3.0, -3.0]) * np.pi
	b = np.array([0.0, -0.5, 1.0, 1.0]) * np.pi
	assert np.allclose(Trafo3d.WrapAngles(a), b)


def GenerateRPYTestCases():
	result = []
	# Important corner cases
	phis = np.linspace(-315.0, 315.0, 15)
	for x_roll in phis:
		for y_pitch in phis:
			for z_yaw in phis:
				result.append(Trafo3d.WrapAngles(np.deg2rad(np.array([ x_roll, y_pitch, z_yaw ]))))
	# Random values
	for i in range(100):
		result.append(2.0 * np.pi * 2.0 * (np.random.rand(3) - 0.5))
	return result


def GenerateTestCases():
	result = []
	for rpy in GenerateRPYTestCases():
		t = 10.0 * 2.0 * (np.random.rand(3) - 0.5)
		mat = RPY2Matrix(rpy)
		hm = np.identity(4)
		hm[0:3,3] = t
		hm[0:3,0:3] = mat
		q = Mat2Quaternion(mat)
		rodr = Quaternion2Rodrigues(q)
		result.append([ t, mat, hm, rpy, q, rodr])
	return result


# Auto-generated test-cases
TestCases = GenerateTestCases()
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
	np.array([ 2.34860312,  0.11097725, -1.81687079])
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
	np.array([ 1.47035792, -1.59311292, -1.5676496 ])
	])


def test_Conversions():
	for t, mat, hm, rpy, q, rodr in TestCases:
		# From Matrix
		trafo = Trafo3d()
		trafo.SetRotationMatrix(mat)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		rpy2 = trafo.GetRotationRPY()
		assert AnglesValid(rpy2)
		assert RPYsEqual(rpy, rpy2)
		q2 = trafo.GetRotationQuaternion()
		assert QuaternionsEqual(q, q2)
		# From homogenous matrix
		trafo = Trafo3d()
		trafo.SetHomogeneousMatrix(hm)
		hm2 = trafo.GetHomogeneousMatrix()
		assert np.allclose(hm, hm2)
		# From RPY
		trafo = Trafo3d()
		trafo.SetRotationRPY(rpy)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		rpy2 = trafo.GetRotationRPY()
		assert AnglesValid(rpy2)
		assert RPYsEqual(rpy, rpy2)
		q2 = trafo.GetRotationQuaternion()
		assert QuaternionsEqual(q, q2)
		rodr2 = trafo.GetRotationRodrigues()
		assert RodriguesEqual(rodr, rodr2)
		# From Quaternion
		trafo = Trafo3d()
		trafo.SetRotationQuaternion(q)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		rpy2 = trafo.GetRotationRPY()
		assert AnglesValid(rpy2)
		assert RPYsEqual(rpy, rpy2)
		q2 = trafo.GetRotationQuaternion()
		assert QuaternionsEqual(q, q2)
		rodr2 = trafo.GetRotationRodrigues()
		assert RodriguesEqual(rodr, rodr2)
		# From Rodrigues
		trafo = Trafo3d()
		trafo.SetRotationRodrigues(rodr)
		mat2 = trafo.GetRotationMatrix()
		assert np.allclose(mat, mat2)
		rpy2 = trafo.GetRotationRPY()
		assert AnglesValid(rpy2)
		assert RPYsEqual(rpy, rpy2)
		q2 = trafo.GetRotationQuaternion()
		assert QuaternionsEqual(q, q2)
		rodr2 = trafo.GetRotationRodrigues()
		assert RodriguesEqual(rodr, rodr2)


def test_Inversions():
	for t, mat, hm, rpy, q, rodr in TestCases:
		trafo = Trafo3d()
		trafo.SetHomogeneousMatrix(hm)
		trafo2 = trafo.Inverse()
		hm2 = trafo2.GetHomogeneousMatrix()
		assert np.allclose(hm, np.linalg.inv(hm2))


def test_MultiplySelf():
	for t, mat, hm, rpy, q, rodr in TestCases:
		trafo = Trafo3d()
		trafo.SetHomogeneousMatrix(hm)
		trafo2 = trafo * trafo.Inverse()
		assert np.allclose(trafo2.GetHomogeneousMatrix(), np.identity(4))
		trafo2 = trafo.Inverse() * trafo
		assert np.allclose(trafo2.GetHomogeneousMatrix(), np.identity(4))


def test_MultiplyTransformations():
	for i in range(len(TestCases) - 1):
		(t1, mat1, hm1, rpy1, q1, rodr1) = TestCases[i]
		(t2, mat2, hm2, rpy2, q2, rodr2) = TestCases[i+1]
		trafo1 = Trafo3d()
		trafo1.SetHomogeneousMatrix(hm1)
		trafo2 = Trafo3d()
		trafo2.SetHomogeneousMatrix(hm2)
		trafo = trafo1 * trafo2
		assert np.allclose(trafo.GetHomogeneousMatrix(), np.dot(hm1, hm2))


def test_MultiplySinglePoints():
	for t, mat, hm, rpy, q, rodr in TestCases:
		trafo = Trafo3d()
		trafo.SetHomogeneousMatrix(hm)
		p = 10.0 * 2.0 * (np.random.rand(3) - 0.5)
		p2 = trafo * p
		ph = np.append(p, 1.0)
		p3 = np.dot(hm, ph)[0:3]
		assert np.allclose(p2, p3)


def test_MultiplyMultiplePoints():
	for t, mat, hm, rpy, q, rodr in TestCases:
		trafo = Trafo3d()
		trafo.SetHomogeneousMatrix(hm)
		n = 2 + np.random.randint(10)
		p = 10.0 * 2.0 * (np.random.rand(n,3) - 0.5)
		p2 = trafo * p
		for i in range(n):
			assert np.allclose(trafo * p[i,:], p2[i,:])


def test_Interpolate():
	t1 = np.array([10, 20, 10])
	r1 = np.deg2rad([30, 0, 0])
	trafo1 = Trafo3d(t=t1, rpy=r1)
	t2 = np.array([20, 10, -20])
	r2 = np.deg2rad([60, 0, 0])
	trafo2 = Trafo3d(t=t2, rpy=r2)

	trafo3 = trafo1.Interpolate(trafo2, weight = 0.0)
	assert np.allclose(t1, trafo3.GetTranslation())
	assert np.allclose(r1, trafo3.GetRotationRPY())

	trafo3 = trafo1.Interpolate(trafo2, weight = 0.5)
	assert np.allclose((t1 + t2) / 2.0, trafo3.GetTranslation())
	assert np.allclose((r1 + r2) / 2.0, trafo3.GetRotationRPY())

	trafo3 = trafo1.Interpolate(trafo2, weight = 1.0)
	assert np.allclose(t2, trafo3.GetTranslation())
	assert np.allclose(r2, trafo3.GetRotationRPY())


def test_InterpolateMultiple():
	trafos = []
	result = Trafo3d.InterpolateMultiple(trafos)
	assert result is None

	t1 = np.array([10, 20, 10])
	r1 = np.deg2rad([90, 0, 0])
	trafo1 = Trafo3d(t=t1, rpy=r1)
	trafos.append(trafo1)
	result = Trafo3d.InterpolateMultiple(trafos)
	assert result == trafo1

	t2 = np.array([20, 10, -20])
	r2 = np.deg2rad([270, 0, 0])
	trafo2 = Trafo3d(t=t2, rpy=r2)
	trafos.append(trafo2)
	result = Trafo3d.InterpolateMultiple(trafos)
	assert result == trafo1.Interpolate(trafo2, weight = 0.5)

	t3 = np.array([100, 110, -120])
	r3 = np.deg2rad([0, 90, 0])
	trafo3 = Trafo3d(t=t3, rpy=r3)
	trafos.append(trafo3)
	result = Trafo3d.InterpolateMultiple(trafos)
	assert result == trafo1.Interpolate(trafo2, weight = 1/2.0).Interpolate(trafo3, weight = 1/3.0)


def test_Distance():
	t1 = np.array([10, 20, 10])
	r1 = np.deg2rad([90, 0, 0])
	trafo1 = Trafo3d(t=t1, rpy=r1)
	dt, dr = trafo1.Distance(trafo1)
	assert dt == 0.0
	assert dr == 0.0

	t2 = np.array([10, 0, 0])
	r2 = np.deg2rad([0, 0, 0])
	trafo2 = Trafo3d(t=t2, rpy=r2)
	dt, dr = trafo2.Distance(trafo2.Inverse())
	assert dt == 20.0
	assert dr == 0.0

	t3 = np.array([0, 0, 0])
	r3 = np.deg2rad([0, 5, 0])
	trafo3 = Trafo3d(t=t3, rpy=r3)
	dt, dr = trafo3.Distance(trafo3.Inverse())
	assert dt == 0.0
	assert np.isclose(np.rad2deg(dr), 10.0)


def test_AverageAndErrors():
	trafos = []
	average, errors = Trafo3d.AverageAndErrors(trafos)
	assert average is None
	assert errors is None

	t1 = np.array([10, 20, 10])
	r1 = np.deg2rad([90, 0, 0])
	trafo1 = Trafo3d(t=t1, rpy=r1)
	trafos.append(trafo1)
	average, errors = Trafo3d.AverageAndErrors(trafos)
	assert average == trafo1
	assert np.allclose(errors, np.zeros(2))

	trafos.append(trafo1)
	average, errors = Trafo3d.AverageAndErrors(trafos)
	assert average == trafo1
	assert np.allclose(errors, np.zeros((2,2)))
