import numpy as np
from trafolib.Trafo3d import Trafo3d



class CameraModel:

	def __init__(self, pix_size, f, c=None, distortion=(0,0,0,0,0), T=Trafo3d()):
		""" Constructor
		:param pix_size: Size of camera chip in pixels (width x height)
		:param f: Focal length, either as scalar f or as vector (fx, fy)
		:param c: Principal point in pixels; if not provided, it is set to pix_size/2
		:param distortion: Parameters for radial (indices 0, 1, 4) and tangential (indices 2, 3) distortion
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
		# distortion parameters
		self.distortion = np.array(distortion)
		if self.distortion.size != 5:
			raise ValueError('Provide 5d distortion vector')
		# camera position: transformation from world to camera
		self.T = T



	def getPixelSize(self):
		return self.pix_size



	def getFocusLength(self):
		return self.f



	def getOpeningAnglesDegrees(self):
		p = np.array([[self.pix_size[0], self.pix_size[1], 1]])
		P = self.chipToScene(p)
		return 2.0 * np.rad2deg(np.arctan2(P[0,0], P[0,2])), \
			2.0 * np.rad2deg(np.arctan2(P[0,1], P[0,2]))



	def getPrincipalPoint(self):
		return self.c



	def getDistortion(self):
		return self.distortion



	def sceneToChip(self, P):
		""" Transforms points in scene to points on chip
		This function does not do any clipping boundary checking!
		:param P: n points P=(X,Y,Z) in scene, shape (n,3)
		:return: n points p=(u,v,d) on chip, shape (n,3)
		"""
		if P.ndim != 2 or P.shape[1] != 3:
			raise ValueError('Provide scene coordinates of shape (n, 3)')
		# Transform points from world coordinate system to camera coordinate system
		P = self.T.Inverse() * P
		# Mask points with Z lesser or equal zero
		valid = P[:,2] > 0.0
		# projection
		x1 = P[valid,0] / P[valid,2]
		y1 = P[valid,1] / P[valid,2]
		# radial distortion
		rsq = x1 * x1 + y1 * y1
		t = 1.0 + self.distortion[0]*rsq + self.distortion[1]*rsq**2 + self.distortion[4]*rsq**3
		x2 = t * x1
		y2 = t * y1
		# tangential distortion
		rsq = x2 * x2 + y2 * y2
		x3 = x2 + 2.0*self.distortion[2]*x2*y2 + self.distortion[3]*(rsq+2*x2*x2)
		y3 = y2 + 2.0*self.distortion[3]*x2*y2 + self.distortion[2]*(rsq+2*y2*y2)
		# focus length and principal point
		p = np.NaN * np.zeros(P.shape)
		p[valid,0] = self.f[0] * x3 + self.c[0]
		p[valid,1] = self.f[1] * y3 + self.c[1]
		p[valid,2] = np.linalg.norm(P[valid,:], axis=1)
		return p



	def scenePointsToDepthImage(self, P, C=None):
		""" Transforms points in scene to depth image
		Image is initialized with np.NaN, invalid chip coordinates are filtered
		:param P: n points P=(X,Y,Z) in scene, shape (n,3)
		:return: Depth image, matrix of shape (self.pix_size[0], self.pix_size[1]), each element is distance
		"""
		p = self.sceneToChip(P)
		# Clip image indices to valid points (can cope with NaN values in p)
		indices = np.round(p[:,0:2]).astype(int)
		x_valid = np.logical_and(indices[:,0] >= 0, indices[:,0] < self.pix_size[0])
		y_valid = np.logical_and(indices[:,1] >= 0, indices[:,1] < self.pix_size[1])
		valid = np.logical_and(x_valid, y_valid)
		# Initialize empty image with NaN
		dImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1]))
		# Set image coordinates to distance values
		dImg[indices[valid,0], indices[valid,1]] = p[valid,2]
		# If color values given, create color image as well
		if C is not None:
			cImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1], 3))
			cImg[indices[valid,0], indices[valid,1], :] = C[valid, :]
			return dImg, cImg
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
		# focus length and principal point
		x3 = (p[:,0] - self.c[0]) / self.f[0]
		y3 = (p[:,1] - self.c[1]) / self.f[1]
		# inverse tangential distortion: TODO
		x2 = x3
		y2 = y3
		# inverse radial distortion
		k1, k2, k3, k4 = self.distortion[[0,1,2,4]]
		# Parameters taken from Pierre Drap: "An Exact Formula for Calculating Inverse Radial Lens Distortions" 2016
		b = np.array([
			-k1,
			3*k1**2 - k2,
			-12*k1**3 + 8*k1*k2 - k3,
			55*k1**4 - 55*k1**2*k2 + 5*k2**2 + 10*k1*k3 - k4,
			-273*k1**5 + 364*k1**3*k2 - 78*k1*k2**2 - 78*k1**2*k3 + 12*k2*k3 + 12*k1*k4,
			1428*k1**6 - 2380*k1**4*k2 + 840*k1**2*k2**2 - 35*k2**3 + 560*k1**3*k3 -210*k1*k2*k3 + 7*k3**2 - 105*k1**2*k4 + 14*k2*k4,
			-7752*k1**7 + 15504*k1**5*k2 - 7752*k1**3*k2**2 + 816*k1*k2**3 - 3876*k1**4*k3 + 2448*k1**2*k2*k3 - 136*k2**2*k3 - 136*k1*k3**2 + 816*k1**3*k4 - 272*k1*k2*k4 + 16*k3*k4,
			43263*k1**8 - 100947*k1**6*k2 + 65835*k1**4*k2**2 - 11970*k1**2*k2**3 + 285*k2**4 + 26334*k1**5*k3 - 23940*k1**3*k2*k3 + 3420*k1*k2**2*k3 + 1710*k1**2*k3**2 - 171*k2*k3**2 - 5985*k1**4*k4 + 3420*k1**2*k2*k4 - 171*k2**2*k4 - 342*k1*k3*k4 + 9*k4**2,
			-246675*k1**9 + 657800*k1**7*k2 - 531300*k1**5*k2**2 + 141680*k1**3*k2**3 - 8855*k1*k2**4 - 177100*k1**6*k3 + 212520*k1**4*k2*k3 - 53130*k1**2*k2**2*k3 + 1540*k2**3*k3 - 17710*k1**3*k3**2 + 4620*k1*k2*k3**2 - 70*k3**3 + 42504*k1**5*k4 - 35420*k1**3*k2*k4 + 4620*k1*k2**2*k4 + 4620*k1**2*k3*k4 - 420*k2*k3*k4 - 210*k1*k4**2
		])
		ssq = x2 * x2 + y2 * y2
		ssqvec = np.array(list(ssq**(i+1) for i in range(b.size)))
		t = 1.0 + np.dot(b, ssqvec)
		x1 = t * x2
		y1 = t * y2
		# projection
		P = np.zeros(p.shape)
		P[:,2] = p[:,2] / np.sqrt(x1*x1 + y1*y2 + 1.0)
		P[:,0] = x1 * P[:,2]
		P[:,1] = y1 * P[:,2]
		# Transform points from camera coordinate system to world coordinate system
		P = self.T * P
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



	@staticmethod
	def __rayIntersectMesh(rayorig, raydir, triangles):
		# Based on Möller–Trumbore intersection algorithm
		# Calculation similar to https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection
		n = triangles.shape[0]
		rays = np.tile(raydir, n).reshape((n,3))
		# Do all calculation no matter if invalid values occur during calculation
		v0 = triangles[:,0,:]
		v0v1 = triangles[:,1,:] - v0
		v0v2 = triangles[:,2,:] - v0
		pvec = np.cross(rays, v0v2, axis=1)
		det = np.sum(np.multiply(v0v1, pvec), axis=1)
		invDet = 1.0 / det
		tvec = rayorig - v0
		u = invDet * np.sum(np.multiply(tvec, pvec), axis=1)
		qvec = np.cross(tvec, v0v1, axis=1)
		v = invDet * np.sum(np.multiply(rays, qvec), axis=1)
		t = invDet * np.sum(np.multiply(v0v2, qvec), axis=1)
		# Check all results for validity
		invalid = np.isclose(det, 0.0)
		invalid = np.logical_or(invalid, u < 0.0)
		invalid = np.logical_or(invalid, u > 1.0)
		invalid = np.logical_or(invalid, v < 0.0)
		invalid = np.logical_or(invalid, (u + v) > 1.0)
		invalid = np.logical_or(invalid, t <= 0.0)
		valid_idx = np.where(~invalid)[0]
		if valid_idx.size == 0:
			# No intersection of ray with any triangle in mesh
			return np.NaN * np.zeros(3), np.NaN * np.zeros(3), -1
		# triangle_index is the index of the triangle intersection point with
		# the lowest z (aka closest to camera); the index is in (0..numTriangles-1)
		Ps = rayorig + raydir[np.newaxis,:] * t[:,np.newaxis]
		triangle_index = valid_idx[Ps[valid_idx,2].argmin()]
		P = Ps[triangle_index,:]
		Pbary = np.array([
			1.0 - u[triangle_index] - v[triangle_index],
			u[triangle_index], v[triangle_index]])
		return P, Pbary, triangle_index



	def __flatShading(mesh, triangle_idx):
		# If we assume a light source behind the camera, the intensity
		# of the triangle (or our point respectively) is the dot product
		# between the normal vector of the triangle and the vector
		# towards the light source [0,0,-1]; this can be simplified:
		normals = mesh.triangle_normals[triangle_idx,:]
		intensities = np.clip(-normals[:,2], 0.0, 1.0)
		return np.vstack((intensities, intensities, intensities)).T



	def __gouraudShading(mesh, P, Pbary, triangle_idx):
		triangles = mesh.triangles[triangle_idx,:]
		vertex_normals = mesh.vertex_normals[triangles]
		n = triangles.shape[0]
		vertex_intensities = np.clip(-vertex_normals[:,:,2], 0.0, 1.0)
		if mesh.vertex_colors is not None:
			vertex_colors = mesh.vertex_colors[triangles]
		else:
			vertex_colors = np.ones((n, 3, 3))
		vertex_color_shades = np.multiply(vertex_colors,
			vertex_intensities[:,:,np.newaxis])
		return np.einsum('ijk,ij->ik', vertex_color_shades, Pbary)



	def snap(self, mesh):
		# Generate camera rays
		rayorig = self.T.GetTranslation()
		img = np.ones((self.pix_size[0], self.pix_size[1]))
		raydir = self.depthImageToScenePoints(img) - rayorig
		# Do raytracing
		P = np.zeros(raydir.shape)
		Pbary = np.zeros(raydir.shape)
		triangle_idx = np.zeros(raydir.shape[0], dtype=int)
		for i in range(raydir.shape[0]):
			P[i,:], Pbary[i,:], triangle_idx[i] = \
				CameraModel.__rayIntersectMesh(rayorig, raydir[i,:], mesh.triangle_vertices)
		# Reduce data to valid intersections of rays with triangles
		valid = ~np.isnan(P[:,0])
		P = P[valid,:]
		Pbary = Pbary[valid,:]
		triangle_idx = triangle_idx[valid]
		# Calculate shading
		#C = CameraModel.__flatShading(mesh, triangle_idx)
		C = CameraModel.__gouraudShading(mesh, P, Pbary, triangle_idx)
		# Determine color and depth images
		dImg, cImg = self.scenePointsToDepthImage(P, C)
		return dImg, cImg, P

