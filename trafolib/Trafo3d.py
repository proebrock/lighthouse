# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import copy
from . pyquaternion import Quaternion


class Trafo3d:

	def __init__(self, *args, **kwargs):
		""" Constructor

		Provide translational or rotatory initializers:

		't'    - Translational, see SetTranslation()
		'mat'  - Rotation matrix, see SetRotationMatrix()
		'hom'  - Homogeneous matrix, see SetHomogeneousMatrix()
		'rpy'  - RPY angles, see SetRotationRPY()
		'q'    - Quaternion, see SetRotationQuaternion()
		'rodr' - Rodrigues rotation formula (OpenCV), see SetRotationRodrigues()
		'list' - Provide transformation as 7 element vector
		         (t and q, as provided by ToList())

		Do not provide multiple translational or multiple rotatory initializers.

		:param args: Non-keyworded fixed position arguments
		:param kwargs: Keyworded arguments
		"""
		if len(args) > 0:
			raise ValueError('No positional arguments allowed')
		if len(frozenset(kwargs.keys()).intersection(set(('t','hom')))) >= 2:
			raise ValueError('Multiple translational components defined')
		if len(frozenset(kwargs.keys()).intersection(set(('mat','hom','rpy','q','rodr')))) >= 2:
			raise ValueError('Multiple rotational components defined')
		if not frozenset(kwargs.keys()).issubset(set(('t','mat','hom','rpy','q','rodr','list'))):
			raise ValueError('Unknown arguments: ' + str(kwargs.keys()))
		self.t = np.zeros(3)
		self.r = Quaternion()
		if 't' in kwargs:
			self.SetTranslation(kwargs['t'])
		if 'mat' in kwargs:
			self.SetRotationMatrix(kwargs['mat'])
		if 'hom' in kwargs:
			self.SetHomogeneousMatrix(kwargs['hom'])
		if 'rpy' in kwargs:
			self.SetRotationRPY(kwargs['rpy'])
		if 'q' in kwargs:
			self.SetRotationQuaternion(kwargs['q'])
		if 'rodr' in kwargs:
			self.SetRotationRodrigues(kwargs['rodr'])
		if 'list' in kwargs:
			if len(kwargs) > 1:
				raise ValueError('If specifying "list", specify no other elements')
			if len(kwargs['list']) != 7:
				raise ValueError('If specifying "list", length has to be 7')
			self.SetTranslation(kwargs['list'][0:3])
			self.SetRotationQuaternion(kwargs['list'][3:])


	def __str__(self):
		""" Get readable string representation of object
		:return: String representation of object
		"""
		tstr = np.array2string(self.t, precision=1, separator=', ', suppress_small=True)
		rpy = np.rad2deg(self.GetRotationRPY())
		rstr = np.array2string(rpy, precision=1, separator=', ', suppress_small=True)
		return '(' + tstr + ', ' + rstr + ')'


	def __repr__(self):
		""" Get unambiguous string representation of object
		:return: String representation of object
		"""
		return repr(self.t) + ',' + repr(self.r.elements)


	def __eq__(self, other):
		""" Check transformations for equality
		:param other: Translation to compare with
		:return: True if transformations are equal, False otherwise
		"""
		if isinstance(other, self.__class__):
			return np.allclose(self.t, other.t) and \
				(np.allclose(self.r.elements, other.r.elements) or \
				np.allclose(self.r.elements, -1.0 * other.r.elements))
		else:
			return False


	def __ne__(self, other):
		""" Check transformations for inequality
		:param other: Translation to compare with
		:return: True if transformations are not equal, False otherwise
		"""
		return not self.__eq__(other)


	def __copy__(self):
		""" Shallow copy
		:return: A shallow copy of self
		"""
		return self.__class__(t=self.t, r=self.r)


	def __deepcopy__(self, memo):
		""" Deep copy
		:param memo: Memo dictionary
		:return: A deep copy of self
		"""
		result = self.__class__(t=copy.deepcopy(self.t, memo), r=copy.deepcopy(self.r, memo))
		memo[id(self)] = result
		return result


	def ToList(self):
		""" Provides transformation as list of values
		Usage is for serialization; deserialize with constructor parameter 'list'
		:return: Transformation as list of values
		"""
		return [ self.t[0], self.t[1], self.t[2], \
			self.r.elements[0], self.r.elements[1], \
			self.r.elements[2], self.r.elements[3] ]


	def Plot2d(self, ax, normal=2, scale=1.0, label=None):
		""" Plotting the tranformation as coordinate system projected into a 2D plane
		:param ax: Axes object created by fig = plt.figure(); ax = fig.add_subplot(111) or similar
		:param normal: Normal vector of plane 0->X-vector->YZ-plane, 1->Y-vector->XZ-plane, 2->Z-vector->XY-plane
		:param scale: Scale factor for axis lengths
		:param label: Label printed close to coordinate system
		"""
		plane = np.arange(3)
		if not normal in plane:
			raise ValueError('Unknown axis: ' + str(normal))
		plane = plane[plane != normal]
		t = self.GetTranslation()
		origin = t[plane]
		m = self.GetRotationMatrix()
		u = m[plane,:]
		colors = ('r', 'g', 'b')
		for i in range(3):
			ui = u[:,i]
			if (np.linalg.norm(ui) > 0.1):
				ax.quiver(*origin, *(ui * scale), color=colors[i],
					angles='xy', scale_units='xy', scale=1.0)
			else:
				c = plt.Circle(origin, 0.15*scale, ec=colors[i], fc='w')
				ax.add_artist(c)
				marker = 'o' if m[normal,i] > 0 else 'x'
				ax.plot(*origin, marker=marker, color=colors[i], markersize=5*scale)
		if label is not None:
			ax.text(*(origin+0.2*scale), label, color='k')


	def PlotAxisAngle(self, ax, scale=1.0, label=None):
		""" Plotting the transformation as rotation axis in 3d
		The axis has a length of 1.0 and is plotted in black; a label denotes the angle in deg
		:param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
		:param scale: Scale factor for axis lengths
		:param label: In None, no label shown, if True angle in deg shown, else given text shown
		"""
		origin = self.GetTranslation()
		axis = self.r.axis * scale
		angle = np.rad2deg(self.r.angle)
		ax.quiver3D(*origin, *axis, color='k', arrow_length_ratio=0.15)
		if label is not None:
			if isinstance(label, bool):
				if label:
					l = f'${np.rad2deg(self.r.angle):.1f}^\circ$'
				else:
					label=''
			else:
				l = label
			ax.text(*(origin+axis+0.1*scale), l, color='k',
				verticalalignment='center', horizontalalignment='center')


	def PlotRodrigues(self, ax, scale=1.0, label=None):
		""" Plotting the transformation as Rodrigues vector in 3d
		The direction and length is determined by the Rodrigues vector
		:param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
		:param scale: Scale factor for axis lengths
		:param label: In None, no label shown, if True angle in deg shown, else given text shown
		"""
		origin = self.GetTranslation()
		axis = self.GetRotationRodrigues() * scale
		ax.quiver3D(*origin, *axis, color='k', arrow_length_ratio=0.15)
		if label is not None:
			if isinstance(label, bool):
				if label:
					l = f'${np.rad2deg(self.r.angle):.1f}^\circ$'
				else:
					label=''
			else:
				l = label
			ax.text(*(origin+axis+0.1*scale), l, color='k',
				verticalalignment='center', horizontalalignment='center')


	def Plot3d(self, ax, scale=1.0, label=None):
		""" Plotting the transformation as coordinate system in 3D
		The axes X/Y/Z each have a length of 1.0 and are plotted in red/green/blue
		:param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
		:param scale: Scale factor for axis lengths
		:param label: Label printed close to coordinate system
		"""
		# Define origin and coordinate axes
		origin = np.zeros(3)
		ux = np.array([scale, 0, 0])
		uy = np.array([0, scale, 0])
		uz = np.array([0, 0, scale])
		# Transform all
		origin = self * origin
		ux = self * ux
		uy = self * uy
		uz = self * uz
		# Plot result and label
		ax.quiver3D(*origin, *(ux-origin), color='r', arrow_length_ratio=0.15)
		ax.quiver3D(*origin, *(uy-origin), color='g', arrow_length_ratio=0.15)
		ax.quiver3D(*origin, *(uz-origin), color='b', arrow_length_ratio=0.15)
		if label is not None:
			l = self * (scale * np.array([0.2, 0.2, 0.2]))
			ax.text(*(l), label, color='k',
				verticalalignment='center', horizontalalignment='center')
			#ax.text(*(origin+0.05), label, color='k')


	def PlotFrustum3d(self, ax, scale=1.0, label=None):
		""" Plotting the transformation as camera frustum in 3D
		The axes X/Y/Z each have a length of 1.0 and are plotted in red/green/blue
		:param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
		:param scale: Scale factor for axis lengths
		:param label: Label printed close to coordinate system
		"""
		# X and Y axes
		origin = np.zeros(3)
		ux = np.array([scale, 0, 0])
		uy = np.array([0, scale, 0])
		origin = self * origin
		ux = self * ux
		uy = self * uy
		ax.quiver3D(*origin, *(ux-origin), color='r')
		ax.quiver3D(*origin, *(uy-origin), color='g')
		# Frustum
		e = 0.2
		z = 1.0
		points = scale * np.array([
			[ 0, 0, 0 ],
			[ -e, -e, z ],
			[ 0, 0, 0 ],
			[ -e,  e, z ],
			[ 0, 0, 0 ],
			[  e,  e, z ],
			[ 0, 0, 0 ],
			[  e, -e, z ],
			[ -e, -e, z ],
			[ -e,  e, z ],
			[ -e,  e, z ],
			[  e,  e, z ],
			[  e,  e, z ],
			[  e, -e, z ],
			[  e, -e, z ],
			[ -e, -e, z ]
			])
		points = self * points
		for i in range(0, points.shape[0], 2):
			ax.plot(points[i:i+2,0], points[i:i+2,1],
				points[i:i+2,2], color='b')
		# Label
		if label is not None:
			ax.text(*(origin+0.05), label, color='k')


	def SetTranslation(self, value):
		""" Set translatory component of transformation
		:param value: Translation as vector (x, y, z)
		"""
		value = np.asarray(value)
		if value.size != 3:
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		self.t = np.reshape(value, (3,))


	def GetTranslation(self):
		""" Get translatory component of transformation
		:return: Translation as vector (x, y, z)
		"""
		return self.t


	def SetRotationMatrix(self, value):
		""" Set rotatory component of transformation as rotation matrix
		:param value: 3x3 rotation matrix
		"""
		value = np.asarray(value)
		if value.shape != (3,3):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		self.r = Quaternion(matrix=value)


	def GetRotationMatrix(self):
		""" Get rotatory component of transformation as rotation matrix
		:return: 3x3 rotation matrix
		"""
		return self.r.rotation_matrix


	def SetHomogeneousMatrix(self, value):
		""" Set translation as homogenous matrix
		:param value: 4x4 homogenous matrix
		"""
		value = np.asarray(value)
		if value.shape != (4,4):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		self.SetTranslation(value[0:3,3])
		self.SetRotationMatrix(value[0:3,0:3])


	def GetHomogeneousMatrix(self):
		""" Get translation as homogenous matrix
		:return: 4x4 homogenous matrix
		"""
		result = np.identity(4)
		result[0:3,3] = self.GetTranslation().reshape((3,))
		result[0:3,0:3] = self.GetRotationMatrix()
		return result


	def SetRotationRPY(self, value):
		""" Set rotatory component of transformation as roll-pitch-yaw angles
		:param value: RPY angles in radians as vector (x_roll, y_pitch, z_yaw)
		"""
		value = np.asarray(value)
		if value.shape != (3,):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		cx = np.cos(value[0])
		cy = np.cos(value[1])
		cz = np.cos(value[2])
		sx = np.sin(value[0])
		sy = np.sin(value[1])
		sz = np.sin(value[2])
		m = np.array([
			[ cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy * cx ],
			[ sz * cy, cz * cx + sz * sy * sx, -cz * sx + sz * sy * cx ],
			[ -sy, cy * sx, cy * cx ]
			])
		self.r = Quaternion(matrix=m)


	@staticmethod
	def WrapAngles(angles):
		""" Wrap any input angle(s) to the range (-pi..pi]
		:param angle: Input angle(s)
		:return: Wrapped angle(s)
		"""
		if isinstance(angles, np.ndarray) or isinstance(angles, list):
			result = np.mod(np.asarray(angles) + np.pi, 2.0 * np.pi) - np.pi
			result[result == -np.pi] = np.pi
		else:
			result = ((angles + np.pi) % (2.0 * np.pi)) - np.pi
			if result == -np.pi:
				result = np.pi
		return result


	def GetRotationRPY(self):
		""" Get rotatory component of transformation as roll-pitch-yaw angles
		:return: RPY angles in radians as vector (x_roll, y_pitch, z_yaw)
		"""
		m = self.r.rotation_matrix
		z = np.arctan2(m[1,0], m[0,0])
		y = np.arctan2(-m[2,0], m[0,0] * np.cos(z) + m[1,0] * np.sin(z))
		x = np.arctan2(m[0,2] * np.sin(z) - m[1,2] * np.cos(z),
			-m[0,1] * np.sin(z) + m[1,1] * np.cos(z));
		result = np.array([ x, y, z ])
		result = Trafo3d.WrapAngles(result)
		return result


	def SetRotationQuaternion(self, value):
		""" Set rotatory component of transformation as unit quarternion
		:param value: Unit quarternion (w, x, y, z)
		"""
		value = np.asarray(value)
		if value.shape != (4,):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		if not np.isclose(np.linalg.norm(value), 1.0):
			raise ValueError('Unit quaternion expected')
		self.r = Quaternion(value)


	def GetRotationQuaternion(self):
		""" Get rotatory component of transformation as unit quarternion
		:return: Unit quarternion (w, x, y, z)
		"""
		return self.r.elements


	def SetRotationRodrigues(self, value):
		""" Set rotatory component of transformation as Rodrigues rotation formula (OpenCV)
		:param value: Rodrigues rotation formula (x, y, z)
		"""
		value = np.asarray(value)
		if value.size != 3:
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		value = np.reshape(value, (3,))
		theta = np.linalg.norm(value)
		if np.isclose(theta, 0.0):
			self.r = Quaternion()
		else:
			axis = value / theta
			self.r = Quaternion(axis=axis, radians=theta)


	def GetRotationRodrigues(self):
		""" Get rotatory component of transformation as Rodrigues rotation formula (OpenCV)
		:return: Rodrigues rotation formula (x, y, z)
		"""
		return self.r.axis * self.r.angle


	def Inverse(self):
		""" Get inverse transformation as a new object
		:return: Inverse transformation
		"""
		r = self.r.inverse
		t = -1.0 * r.rotate(self.t)
		result = self.__class__()
		result.SetTranslation(t)
		result.SetRotationQuaternion(r.elements)
		return result


	def __mul__(self, other):
		""" Multiplication of transformations with other transformation or point(s)

		Sequence of transformations:
		Trafo3d = Trafo3d * Trafo3d

		Transformation of single point
		numpy array size 3 = Trafo3d * numpy array size 3

		Transformation of matrix of points (points as column vectors)
		numpy array size Nx3 = Trafo3d * numpy array size Nx3

		:param other: Transformation or point(s)
		:return: Resulting point of transformation
		"""
		if isinstance(other, Trafo3d):
			result = self.__class__()
			result.t = self.t.reshape((3,)) + self.r.rotate(other.t).reshape((3,))
			result.r = self.r * other.r
			return result
		elif isinstance(other, np.ndarray) or isinstance(other, list):
			other = np.asarray(other)
			if other.size == 3:
				other = np.reshape(other, (3,1))
				return np.reshape(self.t, (3,)) + self.r.rotate(other)
			else:
				if other.ndim == 2 and (other.shape[1] != 3):
					raise ValueError('Second dimension must be 3')
				t = np.tile(self.t, (other.shape[0], 1))
				r = np.dot(other, self.r.rotation_matrix.T)
				return t + r
		else:
			raise ValueError('Expecting instance of Trafo3d or numpy array')


	@staticmethod
	def QuaternionSlerp(q0, q1, weight):
		# Taken from https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
		v0 = q0.elements
		v0 = v0 / np.linalg.norm(v0)
		v1 = q1.elements
		v1 = v1 / np.linalg.norm(v1)
		weight = np.clip(weight, 0, 1)

		# This block ensures the shortest path quaternion slerp
		dot = np.dot(v0, v1)
		if dot < 0.0:
			v1 = -v1
			dot = -dot

		# If inputs are too close for comfort: interpolate linearily
		if dot > 0.9999995:
			result = v0 +  weight * (v1 - v0)
			result = result / np.linalg.norm(result)
			return Quaternion(array=result)

		theta0 = np.arccos(dot)
		theta = theta0 * weight
		sin_theta = np.sin(theta)
		sin_theta0 = np.sin(theta0)
		s0 = np.cos(theta) - dot * sin_theta / sin_theta0
		s1 = sin_theta / sin_theta0
		return Quaternion(array=(s0 * v0) + (s1 * v1))


	def Interpolate(self, other, weight=0.5, shortest_path=True):
		""" Interpolate two transformations
		Uses Slerps (spherical linear interpolation) for rotatory component
		:param other: Second transformation
		:param weight: Weight of second transformation [0..1];
			if weight is 0.0, the result is self,
			if weight is 1.0, the result is other
		:return: Resulting interpolated transformation
		"""
		if (weight < 0.0) or (weight > 1.0):
			raise ValueError('Invalid weight')
		result = self.__class__()
		result.t = (1.0 - weight) * self.t + weight * other.t
		if shortest_path:
			result.r = Trafo3d.QuaternionSlerp(self.r, other.r, weight)
		else:
			result.r = Quaternion.slerp(self.r, other.r, weight)
		return result


	@staticmethod
	def InterpolateMultiple(trafos, shortest_path=True):
		""" Interpolate an array of transformations
		Uses Slerps (spherical linear interpolation) for rotatory component
		:param trafos: List of transformations
		:return: Resulting interpolated transformation
		"""
		if len(trafos) == 0:
			return None
		result = trafos[0]
		for i, t in enumerate(trafos[1:]):
			result = result.Interpolate(t, weight = 1.0 / (i + 2), shortest_path=shortest_path)
		return result


	def Distance(self, other):
		""" Calculate difference between two transformations:
		For translational component - Euclidian distance
		For rotatory component - Absolute of rotation angle around rotation axis (quaternion)
		"""
		delta = self.Inverse() * other
		dt = np.linalg.norm(delta.t)
		dr = np.abs(delta.r.angle)
		#dt = np.linalg.norm(self.t - other.t)
		#dr = Quaternion.absolute_distance(self.r, other.r)
		return dt, dr


	@staticmethod
	def AverageAndErrors(trafos):
		"""Interpolate an array of transformations and return average and errors
		Average uses Slerps (spherical linear interpolation) for rotatory component
		Each row of error matrix contains Trafo3d.Distance of average and trafo
		Error matrix size: len(trafos) x 2
		"""
		if len(trafos) == 0:
			return None, None
		average = Trafo3d.InterpolateMultiple(trafos, True)
		errors = np.zeros((len(trafos), 2))
		for i, t in enumerate(trafos):
			dt, dr = average.Distance(t)
			errors[i,:] = [ dt, dr ]
		return average, errors

