# -*- coding: utf-8 -*-


import numpy as np


class Trafo2d:

	def __init__(self, *args, **kwargs):
		""" Constructor

		Provide translational or rotatory initializers:

		't'   - Translational, see SetTranslation()
		'mat' - Rotation matrix, see SetRotationMatrix()
		'hom' - Homogeneous matrix, see SetHomogeneousMatrix()
		'angle' - Rotation angle, see SetRotationAngle()

		Do not provide multiple translational or multiple rotatory initializers.

		:param args: Non-keyworded fixed position arguments
		:param kwargs: Keyworded arguments
		"""
		if len(args) > 0:
			raise ValueError('No positional arguments allowed')
		if len(frozenset(kwargs.keys()).intersection(set(('t','hom')))) >= 2:
			raise ValueError('Multiple translational components defined')
		if len(frozenset(kwargs.keys()).intersection(set(('mat','hom','angle')))) >= 2:
			raise ValueError('Multiple rotational components defined')
		if not frozenset(kwargs.keys()).issubset(set(('t','mat','hom','angle'))):
			raise ValueError('Unknown arguments: ' + str(kwargs.keys()))
		self.hom = np.identity(3)
		if 't' in kwargs:
			self.SetTranslation(kwargs['t'])
		if 'mat' in kwargs:
			self.SetRotationMatrix(kwargs['mat'])
		if 'hom' in kwargs:
			self.SetHomogeneousMatrix(kwargs['hom'])
		if 'angle' in kwargs:
			self.SetRotationAngle(kwargs['angle'])


	def __str__(self):
		""" Get readable string representation of object
		:return: String representation of object
		"""
		tstr = np.array2string(self.GetTranslation(), precision=1, separator=', ', suppress_small=True)
		angle = np.rad2deg(self.GetRotationAngle())
		anglestr = np.array2string(angle, precision=1, separator=', ', suppress_small=True)
		return '(' + tstr + ', ' + anglestr + ')'


	def __repr__(self):
		""" Get unambiguous string representation of object
		:return: String representation of object
		"""
		return repr(self.hom)


	def __eq__(self, other):
		""" Check transformations for equality
		:param other: Translation to compare with
		:return: True if transformations are equal, False otherwise
		"""
		if isinstance(other, self.__class__):
			return np.allclose(self.hom, other.hom)
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
		return self.__class__(hom=self.hom)


	def __deepcopy__(self, memo):
		""" Deep copy
		:param memo: Memo dictionary
		:return: A deep copy of self
		"""
		result = self.__class__(hom=copy.deepcopy(self.hom, memo))
		memo[id(self)] = result
		return result

	
	def Plot2d(self, ax, scale=1.0, label=None):
		""" Plotting the transformation as coordinate system
		The axes X/Y each have a length of 1.0 and are plotted in red/green
		:param ax: Axes object
		:param scale: Scale factor for axis lengths
		:param label: Optional label printed close to coordinate system
		"""
		# Define origin and coordinate axes
		origin = np.zeros(2)
		ux = np.array([scale, 0])
		uy = np.array([0, scale])
		# Transform all
		origin = self * origin
		ux = self * ux
		uy = self * uy
		# Plot result and label
#		ax.arrow(*origin, *(ux-origin), length_includes_head=True,
#			head_width=scale/4, head_length=scale/4, color='r')
#		ax.arrow(*origin, *(uy-origin), length_includes_head=True,
#			head_width=scale/4, head_length=scale/4, color='g')
		ax.quiver(*origin, *(ux-origin), color='r',
			angles='xy', scale_units='xy', scale=1.0)
		ax.quiver(*origin, *(uy-origin), color='g',
			angles='xy', scale_units='xy', scale=1.0)
		if label is not None:
			l = self * (scale * np.array([0.4, 0.4]))
			ax.text(*(l), label, color='k',
				verticalalignment='center', horizontalalignment='center')


	def SetTranslation(self, value):
		""" Set translatory component of transformation
		:param value: Translation as vector (x, y)
		"""
		value = np.asarray(value)
		if value.size != 2:
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		self.hom[0:2,2] = value


	def GetTranslation(self):
		""" Get translatory component of transformation
		:return: Translation as vector (x, y)
		"""
		return self.hom[0:2,2]


	def SetRotationMatrix(self, value):
		""" Set rotatory component of transformation as rotation matrix
		:param value: 2x2 rotation matrix
		"""
		value = np.asarray(value)
		if value.shape != (2,2):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		if not np.allclose(np.dot(value, value.conj().transpose()), np.eye(2)):
			raise ValueError("Matrix must be orthogonal, i.e. its transpose should be its inverse")
		if not np.isclose(np.linalg.det(value), 1.0):
			raise ValueError("Matrix must be special orthogonal i.e. its determinant must be +1.0")
		self.hom[0:2,0:2] = value


	def GetRotationMatrix(self):
		""" Get rotatory component of transformation as rotation matrix
		:return: 2x2 rotation matrix
		"""
		return self.hom[0:2,0:2]


	def SetHomogeneousMatrix(self, value):
		""" Set translation as homogenous matrix
		:param value: 3x3 homogenous matrix
		"""
		value = np.asarray(value)
		if value.shape != (3,3):
			raise ValueError('Initialization with invalid shape: ', str(value.shape))
		self.hom = value


	def GetHomogeneousMatrix(self):
		""" Get translation as homogenous matrix
		:return: 3x3 homogenous matrix
		"""
		return self.hom


	def SetRotationAngle(self, angle):
		""" Set rotatory component of transformation as rotation angles
		:param angle: Rotation angle
		"""
		c = np.cos(angle)
		s = np.sin(angle)
		self.hom[0:2,0:2] = np.array([ [ c, -s ], [ s, c ] ])


	@staticmethod
	def WrapAngle(angle):
		""" Wrap any input angle to the range (-pi..pi]
		:param angle: Input angle
		:return: Wrapped angle
		"""
		result = ((angle + np.pi) % (2 * np.pi)) - np.pi
		if result == -np.pi:
			result = np.pi
		return result


	def GetRotationAngle(self):
		""" Get rotatory component of transformation as rotation angle
		:return: Rotation angle
		"""
		return np.arctan2(self.hom[1,0], self.hom[0,0])


	def Inverse(self):
		""" Get inverse transformation as a new object
		:return: Inverse transformation
		"""
		result = self.__class__()
		result.SetHomogeneousMatrix(np.linalg.inv(self.hom))
		return result


	def __mul__(self, other):
		""" Multiplication of transformations with other transformation or point(s)

		Sequence of transformations:
		Trafo2d = Trafo2d * Trafo2d

		Transformation of single point
		numpy array size 2 = Trafo2d * numpy array size 2

		Transformation of matrix of points (points as column vectors)
		numpy array size 2xN = Trafo2d * numpy array size 2xN

		:param other: Transformation or point(s)
		:return: Resulting point of transformation
		"""
		if isinstance(other, Trafo2d):
			result = self.__class__()
			result.hom = np.dot(self.hom, other.hom)
			return result
		elif isinstance(other, np.ndarray) or isinstance(other, list):
			other = np.asarray(other)
			if other.size == 2:
				other = np.reshape(other, (2,1))
				t = np.reshape(self.hom[0:2,2], (2,1))
				return np.reshape(t + np.dot(self.hom[0:2,0:2], other), (2,))
			else:
				if other.ndim == 2 and (other.shape[1] != 2):
					raise ValueError('Second dimension must be 2')
				t = np.tile(self.hom[0:2,2].T, (other.shape[0],1))
				r = np.dot(other, self.hom[0:2,0:2].T)
				return t + r
		else:
			raise ValueError('Expecting instance of Trafo2d or numpy array')

