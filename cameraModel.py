import numpy as np



class CameraModel:
	
	def __init__(self, pix_size, f, c=None):
		""" Constructor
		:param pix_size: Size of camera chip in pixels (width x height)
		:param f: Focal length, either as scalar f or as vector (fx, fy)
		:param c: Principal point in pixels; if not provided, it is set to pix_size/2
		"""
		# pix_size
		self.pix_size = np.asarray(pix_size)
		if self.pix_size.size != 2:
			raise ValueError('Provide 2d chip size in pixels')
		if np.any(self.pix_size < 1):
			raise ValueError('Provide positive chip size')
		# f
		self.f = np.asarray(f)
		if self.f.size == 1:
			self.f = np.append(self.f, self.f)
		elif self.f.size > 2:
			raise ValueError('Provide 1d or 2d focal length')
		if np.any(self.f < 0) or np.any(np.isclose(self.f, 0)):
			raise ValueError('Provide positive focal length')
		# c
		if c is not None:
			self.c = np.asarray(c)
			if self.c.size != 2:
				raise ValueError('Provide 2d principal point')
		else:
			self.c = self.pix_size / 2.0



	def getOpeningAnglesDegrees(self):
		p = np.array([[self.pix_size[0], self.pix_size[1], 1]])
		P = self.chipToScene(p)
		return 2.0 * np.rad2deg(np.arctan2(P[0,0], P[0,2])), \
			2.0 * np.rad2deg(np.arctan2(P[0,1], P[0,2]))



	def sceneToChip(self, P):
		""" Transforms points in scene to points on chip
		This function does not do any clipping boundary checking!
		:param P: n points P=(X,Y,Z) in scene, shape (n,3)
		:return: n points p=(u,v,d) on chip, shape (n,3)
		"""
		if P.ndim != 2 or P.shape[1] != 3:
			raise ValueError('Provide scene coordinates of shape (n, 3)')
		if np.any(P[:,2] < 0) or np.any(np.isclose(P[:,2], 0)):
			raise ValueError('Z coordinate must be greater than zero')
		p = np.zeros(P.shape)
		p[:,0] = (self.f[0] * P[:,0]) / P[:,2] + self.c[0]
		p[:,1] = (self.f[1] * P[:,1]) / P[:,2] + self.c[1]
		p[:,2] = np.linalg.norm(P, axis=1)
		return p



	def scenePointsToDepthImage(self, P, I=None):
		""" Transforms points in scene to depth image
		Image is initialized with np.NaN, invalid chip coordinates are filtered
		:param P: n points P=(X,Y,Z) in scene, shape (n,3)
		:return: Depth image, matrix of shape (self.pix_size[0], self.pix_size[1]), each element is distance
		"""
		p = self.sceneToChip(P)
		# Clip image indices to valid points
		indices = np.round(p[:,0:2]).astype(int)
		x_valid = np.logical_and(indices[:,0] >= 0, indices[:,0] < self.pix_size[0])
		y_valid = np.logical_and(indices[:,1] >= 0, indices[:,1] < self.pix_size[1])
		valid = np.logical_and(x_valid, y_valid)
		# Initialize empty image with NaN
		dImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1]))
		# Set image coordinates to distance values
		dImg[indices[valid,0], indices[valid,1]] = p[valid,2]
		# If intensity values given, create intensity image as well
		if I is not None:
			iImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1]))
			iImg[indices[valid,0], indices[valid,1]] = I[valid]
			return dImg, iImg
		else:
			return dImg



	def chipToScene(self, p):
		""" Transforms points on chip to points in scene
		This function does not do any clipping boundary checking!
		:param p: n points p=(u,v,d) on chip, shape (n,3)
		:return: n points P=(X,Y,Z) in scene, shape (n,3)
		"""
		if p.ndim != 2 or p.shape[1] != 3:
			raise ValueError('Provide chip coordinates of shape (n, 3)')
		P = np.zeros(p.shape)
		P[:,2] = p[:,2] / np.sqrt(
			np.square((p[:,0]-self.c[0])/self.f[0]) +
			np.square((p[:,1]-self.c[1])/self.f[1]) +
			1)
		P[:,0] = ((p[:,0]-self.c[0])*P[:,2])/self.f[0]
		P[:,1] = ((p[:,1]-self.c[1])*P[:,2])/self.f[1]
		return P



	def depthImageToScenePoints(self, img):
		""" Transforms depth image to list of scene points
		:param img: Depth image, matrix of shape (self.pix_size[0], self.pix_size[1]), each element is distance or NaN
		:return: n points P=(X,Y,Z) in scene, shape (n,3) with n=np.prod(self.pix_size) - number of NaNs
		"""
		if not np.all(np.equal(self.pix_size, img.shape)):
			raise ValueError('Provide depth image of proper size')
		mask = ~np.isnan(img)
		if not np.all(img[mask] >= 0.0):
			raise ValueError('Depth image must contain only positive distances or NaN')
		x = np.arange(self.pix_size[0])
		y = np.arange(self.pix_size[1])
		X, Y = np.meshgrid(x, y, indexing='ij')
		p = np.vstack((X.flatten(), Y.flatten(), img.flatten())).T
		mask = np.logical_not(np.isnan(p[:,2]))
		return self.chipToScene(p[mask])



	def getCameraRays(self):
		img = np.ones((self.pix_size[0], self.pix_size[1]))
		return self.depthImageToScenePoints(img)



	@staticmethod
	def __rayIntersectTriangle(ray, triangle):
		# Based on Möller–Trumbore intersection algorithm
		v0 = triangle[0,:]
		e1 = triangle[1,:] - v0
		e2 = triangle[2,:] - v0
		h = np.cross(ray, e2)
		a = np.dot(e1, h)
		if np.isclose(a, 0.0):
			return None
		f = 1.0 / a
		u = -f * np.dot(v0, h)
		if (u < 0.0) or (u > 1.0):
			return None
		q = np.cross(e1, v0)
		v = f * np.dot(ray, q)
		if (v < 0.0) or ((u + v) > 1.0):
			return None
		t = f * np.dot(e2, q)
		if t <= 0.0:
			return None
		return ray * t

	
	@staticmethod
	def __rayIntersectMesh_slow(ray, mesh):
		vertices = np.asarray(mesh.vertices)
		triangles = np.asarray(mesh.triangles)
		result = np.NaN * np.zeros(3)
		for t in triangles:
			P = CameraModel.__rayIntersectTriangle(ray, vertices[t])
			if P is not None:
				if np.isnan(result[2]) or (P[2] < result[2]):
					result = P
		return result



	@staticmethod
	def __rayIntersectMesh(ray, mesh):
		vertices = np.asarray(mesh.vertices)
		triangles = vertices[np.asarray(mesh.triangles)]
		n = triangles.shape[0]
		rays = np.tile(ray, n).reshape((n,3))
		# Do all calculation no matter if invalid values occur during calculation
		v0 = triangles[:,0,:]
		e1 = triangles[:,1,:] - v0
		e2 = triangles[:,2,:] - v0
		h = np.cross(rays, e2, axis=1)
		a = np.sum(np.multiply(e1, h), axis=1)
		f = 1.0 / a
		u = -f * np.sum(np.multiply(v0, h), axis=1)
		q = np.cross(e1, v0, axis=1)
		v = f * np.sum(np.multiply(rays, q), axis=1)
		t = f * np.sum(np.multiply(e2, q), axis=1)
		# Check all results for validity
		invalid = np.isclose(a, 0.0)
		invalid = np.logical_or(invalid, u < 0.0)
		invalid = np.logical_or(invalid, u > 1.0)
		invalid = np.logical_or(invalid, v < 0.0)
		invalid = np.logical_or(invalid, (u + v) > 1.0)
		invalid = np.logical_or(invalid, t <= 0.0)
		# Calculate valid results
		Ps = ray[np.newaxis,:] * t[~invalid,np.newaxis]
		if Ps.size == 0:
			return np.NaN * np.zeros(3), np.NaN
		else:
			z_min_index = np.nanargmin(Ps[:,2])
			P = Ps[z_min_index, :]
			normals = np.asarray(mesh.triangle_normals)[~invalid]
			# Flat shading:
			# If we assume a light source behind the camera, the intensity
			# of the triangle (or our point respectively) is the dot product
			# between the normal vector of the triangle and the vector
			# towards the light source [0,0,-1]; this can be simplified:
			I = -normals[z_min_index,2]
			return P, I



	def snap(self, mesh):
		rays = self.getCameraRays()
		P = np.zeros(rays.shape)
		I = np.zeros(rays.shape[0])
		for i in range(rays.shape[0]):
			P[i,:], I[i] = CameraModel.__rayIntersectMesh(rays[i,:], mesh)
		invalid = np.isnan(I)
		P = P[~invalid,:]
		I = I[~invalid]
		return self.scenePointsToDepthImage(P, I)

