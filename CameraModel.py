import numpy as np
from trafolib.Trafo3d import Trafo3d



class CameraModel:
    """ Class for simulating a depth and/or RGB camera
    """

    def __init__(self, pix_size, f, c=None, distortion=(0, 0, 0, 0, 0), T=Trafo3d(), shadingMode='gouraud'):
        """ Constructor
        :param pix_size: Size of camera chip in pixels (width x height)
        :param f: Focal length, either as scalar f or as vector (fx, fy)
        :param c: Principal point in pixels; if not provided, it is set to pix_size/2
        :param distortion: Parameters (k1, k2, p1, p2, k3) for radial (kx) and
            tangential (px) distortion
        :param T: Transformation from world coordinate system to camera coordinate system
        :param shadingMode: Shading mode, 'flat' or 'gouraud'
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
        # shading mode
        if shadingMode not in ('flat', 'gouraud'):
            raise ValueError(f'Unknown shading mode "{shadingMode}')
        self.shadingMode = shadingMode



    def getPixelSize(self):
        """ Get pixel size
        :returns: Size of camera chip in pixels (width x height)
        """
        return self.pix_size



    def getFocusLength(self):
        """ Get focus length
        :returns: Focus lengths (fx, fy)
        """
        return self.f



    def getOpeningAnglesDegrees(self):
        """ Calculate opening angles
        :returns: Opening angles in x and y in degrees
        """
        p = np.array([[self.pix_size[0], self.pix_size[1], 1]])
        P = self.chipToScene(p)
        return 2.0 * np.rad2deg(np.arctan2(P[0, 0], P[0, 2])), \
            2.0 * np.rad2deg(np.arctan2(P[0, 1], P[0, 2]))



    def getPrincipalPoint(self):
        """ Get principal point
        :returns: Coordinates of principal point (cx, cy)
        """
        return self.c



    def getDistortion(self):
        """ Get distortion parameters
        :returns: Parameters (k1, k2, p1, p2, k3) for radial (kx) and tangential (px) distortion
        """
        return self.distortion



    def getCameraPosition(self):
        """ Get camera position
        :returns: Transformation from world coordinate system to camera coordinate system
        """
        return self.T



    def sceneToChip(self, P):
        """ Transforms points in scene to points on chip
        This function does not do any clipping boundary checking!
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :returns: n points p=(u, v, d) on chip, shape (n, 3)
        """
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError('Provide scene coordinates of shape (n, 3)')
        # Transform points from world coordinate system to camera coordinate system
        P = self.T.Inverse() * P
        # Mask points with Z lesser or equal zero
        valid = P[:, 2] > 0.0
        # projection
        x1 = P[valid, 0] / P[valid, 2]
        y1 = P[valid, 1] / P[valid, 2]
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
        p[valid, 0] = self.f[0] * x3 + self.c[0]
        p[valid, 1] = self.f[1] * y3 + self.c[1]
        p[valid, 2] = np.linalg.norm(P[valid, :], axis=1)
        return p



    def scenePointsToDepthImage(self, P, C=None):
        """ Transforms points in scene to depth image
        Image is initialized with np.NaN, invalid chip coordinates are filtered
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :param C: n colors C=(R, G, B) for each point; same shape as P; optional
        :returns: Depth image, matrix of shape (self.pix_size[0], self.pix_size[1]),
            each element is distance; if C was provided, also returns color image
            of same size
        """
        p = self.sceneToChip(P)
        # Clip image indices to valid points (can cope with NaN values in p)
        indices = np.round(p[:, 0:2]).astype(int)
        x_valid = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < self.pix_size[0])
        y_valid = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < self.pix_size[1])
        valid = np.logical_and(x_valid, y_valid)
        # Initialize empty image with NaN
        dImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1]))
        # Set image coordinates to distance values
        dImg[indices[valid, 0], indices[valid, 1]] = p[valid, 2]
        # If color values given, create color image as well
        if C is not None:
            if not np.array_equal(P.shape, C.shape):
                raise ValueError('P and C have to have the same shape')
            cImg = np.NaN * np.empty((self.pix_size[0], self.pix_size[1], 3))
            cImg[indices[valid, 0], indices[valid, 1], :] = C[valid, :]
            return dImg, cImg
        return dImg



    def chipToScene(self, p):
        """ Transforms points on chip to points in scene
        This function does not do any clipping boundary checking!
        :param p: n points p=(u, v, d) on chip, shape (n, 3)
        :returns: n points P=(X, Y, Z) in scene, shape (n, 3)
        """
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Provide chip coordinates of shape (n, 3)')
        # focus length and principal point
        x3 = (p[:, 0] - self.c[0]) / self.f[0]
        y3 = (p[:, 1] - self.c[1]) / self.f[1]
        # inverse tangential distortion: TODO
        x2 = x3
        y2 = y3
        # inverse radial distortion
        k1, k2, k3, k4 = self.distortion[[0, 1, 2, 4]]
        # Parameters taken from Pierre Drap: "An Exact Formula
        # for Calculating Inverse Radial Lens Distortions" 2016
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
        P[:, 2] = p[:, 2] / np.sqrt(x1*x1 + y1*y2 + 1.0)
        P[:, 0] = x1 * P[:, 2]
        P[:, 1] = y1 * P[:, 2]
        # Transform points from camera coordinate system to world coordinate system
        P = self.T * P
        return P



    def depthImageToScenePoints(self, img):
        """ Transforms depth image to list of scene points
        :param img: Depth image, matrix of shape (self.pix_size[0], self.pix_size[1]),
            each element is distance or NaN
        :returns: n points P=(X, Y, Z) in scene, shape (n, 3) with
            n=np.prod(self.pix_size) - number of NaNs
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
        mask = np.logical_not(np.isnan(p[:, 2]))
        return self.chipToScene(p[mask])



    @staticmethod
    def __rayIntersectMesh(rayorig, raydir, triangles):
        """ Intersection of an ray with a number of triangles
        Tests intersection of ray with all triangles and returns the one with lowest Z coordinate
        Based on Möller–Trumbore intersection algorithm (see https://scratchapixel.com)
        :param rayorig: Ray origin, size 3 (X, Y, Z)
        :param raydir: Ray direction, size 3 (X, Y, Z)
        :param triangles: Triangles, shape (n, 3, 3) - (num triangles, num vertices, XYZ)
        :returns:
            - P - Intersection point of ray with triangle in Cartesian
                  coordinates (X, Y, Z) or (NaN, NaN, NaN)
            - Pbary - Intersection point of ray within triangle in barycentric
                  coordinates (1-u-v, u, v) or (NaN, NaN, NaN)
            - triangle_index - Index of triangle intersecting with ray (0..n-1) or -1
        """
        n = triangles.shape[0]
        rays = np.tile(raydir, n).reshape((n, 3))
        # Do all calculation no matter if invalid values occur during calculation
        v0 = triangles[:, 0, :]
        v0v1 = triangles[:, 1, :] - v0
        v0v2 = triangles[:, 2, :] - v0
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
        # the lowest t, which means it is the intersection point closest to the camera
        triangle_index = valid_idx[t[valid_idx].argmin()]
        P = rayorig + raydir * t[triangle_index]
        Pbary = np.array([
            1.0 - u[triangle_index] - v[triangle_index],
            u[triangle_index], v[triangle_index]])
        return P, Pbary, triangle_index



    @staticmethod
    def __flatShading(mesh, triangle_idx, lightvec):
        """ Calculate flat shading for multiple triangles
        If the angle between the normal vector of the triangle and lightvec
        is 0, the intensity of the triangle is 1.0, for 90 degrees or more it
        is 0.0; the intensity is the dot product between both vectors
        Assumption: Position of camera and light source are identical
        :param mesh: Mesh of type MeshObject
        :param triangle_idx: Indices of n triangles whose shading we want to calculate, shape (n, )
        :param lightvec: Unit vector pointing towards the camera and the light source
        :returns: Shades of triangles; shape (n, 3) (RGB) [0.0..1.0]
        """
        normals = mesh.triangle_normals[triangle_idx, :]
        intensities = np.clip(np.dot(normals, lightvec), 0.0, 1.0)
        return np.vstack((intensities, intensities, intensities)).T



    @staticmethod
    def __gouraudShading(mesh, Pbary, triangle_idx, lightvec):
        """ Calculate the Gouraud shading for multiple points
        For each vertex of the triangle, it calculates the intensity from
        vertex normal and lightvec and uses this to determine the color of
        the vertex. Finally, the color at the point is interpolated using
        its barycentric coordinates
        Assumption: Position of camera and light source are identical
        :param mesh: Mesh of type MeshObject
        :param P: Points on triangles in Cartesian coordinates (X, Y, Z), shape (n, 3)
        :param Pbary: Points on triangles in barycentric coordinates (1-u-v, u, v), shape (n, 3)
        :param triangle_idx: Indices of n triangles whose shading we want to calculate, shape (n, )
        :param lightvec: Unit vector pointing towards the camera and the light source
        :returns: Shades of triangles; shape (n, 3) (RGB) [0.0..1.0]
        """
        triangles = mesh.triangles[triangle_idx, :]
        vertex_normals = mesh.vertex_normals[triangles]
        n = triangles.shape[0]
        vertex_intensities = np.clip(np.dot(vertex_normals, lightvec), 0.0, 1.0)
        if mesh.vertex_colors is not None:
            vertex_colors = mesh.vertex_colors[triangles]
        else:
            vertex_colors = np.ones((n, 3, 3))
        vertex_color_shades = np.multiply(vertex_colors,
                                          vertex_intensities[:, :, np.newaxis])
        return np.einsum('ijk, ij->ik', vertex_color_shades, Pbary)



    def snap(self, mesh):
        """ Takes image of mesh using camera
        :returns:
            - dImg - Depth image of scene, pixels seeing no object are set to NaN
            - cImg - Color image (RGB) of scene, pixels seeing no object are set to NaN
            - P - Scene points (only valid points)
        """
        # Generate camera rays
        rayorig = self.T.GetTranslation()
        img = np.ones((self.pix_size[0], self.pix_size[1]))
        raydir = self.depthImageToScenePoints(img) - rayorig
        # Do raytracing
        P = np.zeros(raydir.shape)
        Pbary = np.zeros(raydir.shape)
        triangle_idx = np.zeros(raydir.shape[0], dtype=int)
        for i in range(raydir.shape[0]):
            P[i, :], Pbary[i, :], triangle_idx[i] = \
                CameraModel.__rayIntersectMesh(rayorig, raydir[i, :], mesh.triangle_vertices)
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(P[:, 0])
        P = P[valid, :]
        Pbary = Pbary[valid, :]
        triangle_idx = triangle_idx[valid]
        # Calculate shading
        lightvec = -self.T.GetRotationMatrix()[:, 2]
        if self.shadingMode == 'flat':
            C = CameraModel.__flatShading(mesh, triangle_idx, lightvec)
        elif self.shadingMode == 'gouraud':
            C = CameraModel.__gouraudShading(mesh, Pbary, triangle_idx, lightvec)
        # Determine color and depth images
        dImg, cImg = self.scenePointsToDepthImage(P, C)
        return dImg, cImg, P
