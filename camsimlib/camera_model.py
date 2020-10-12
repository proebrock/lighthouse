import json
import numpy as np
from trafolib.trafo3d import Trafo3d



class CameraModel:
    """ Class for simulating a depth and/or RGB camera

    Camera coordinate system with Z-Axis pointing into direction of view

    Z               X - Axis
                    self.chip_size[0] = width
      X--------->   dImg second dimension
      |
      |
      |
      |
      V

    Y - Axis
    self.chip_size[1] = height
    dImg first dimension

    """

    def __init__(self, chip_size, focal_length, principal_point=None,
                 distortion=(0, 0, 0, 0, 0), camera_position=Trafo3d(),
                 shading_mode='gouraud'):
        """ Constructor
        :param chip_size: See set_chip_size()
        :param focal_length: See set_focal_length()
        :param principal_point: See set_principal_point(); if not provided, it is set center of chip
        :param distortion: See set_distortion()
        :param camera_position: See set_camera_position()
        :param shading_mode: Shading mode, 'flat' or 'gouraud'
        """
        # chip_size
        self.chip_size = None
        self.set_chip_size(chip_size)
        # focal_length
        self.focal_length = None
        self.set_focal_length(focal_length)
        # principal_point
        self.principal_point = None
        if principal_point is not None:
            self.set_principal_point(principal_point)
        else:
            self.set_principal_point(self.chip_size / 2.0)
        # distortion
        self.distortion = None
        self.set_distortion(distortion)
        # camera position: transformation from world to camera
        self.camera_position = None
        self.set_camera_position(camera_position)
        # shading mode
        if shading_mode not in ('flat', 'gouraud'):
            raise ValueError(f'Unknown shading mode "{shading_mode}')
        self.shading_mode = shading_mode



    def __str__(self):
        """ String representation of camera object
        :returns: String representing camera object
        """
        return (f'chip_size={self.chip_size}, '
                f'f={self.focal_length}, '
                f'c={self.principal_point}, '
                f'distortion={self.distortion}, '
                f'camera_position={self.camera_position}')



    def set_chip_size(self, chip_size):
        """ Set chip size
        The size of the camera chip in pixels, width x height
        :param chip_size: Chip size
        """
        csize = np.asarray(chip_size, dtype=np.int64)
        if csize.size != 2:
            raise ValueError('Provide 2d chip size in pixels')
        if np.any(csize < 1):
            raise ValueError('Provide positive chip size')
        self.chip_size = csize



    def get_chip_size(self):
        """ Get chip size
        The size of the camera chip in pixels, width x height
        :returns: Chip size
        """
        return self.chip_size



    def set_focal_length(self, focal_length):
        """ Set focal length
        Focal length, either as scalar f or as vector (fx, fy)
        :param focal_length: Focal length
        """
        flen = np.asarray(focal_length)
        if flen.size == 1:
            flen = np.append(flen, flen)
        elif flen.size > 2:
            raise ValueError('Provide 1d or 2d focal length')
        if np.any(flen < 0) or np.any(np.isclose(flen, 0)):
            raise ValueError('Provide positive focal length')
        self.focal_length = flen



    def get_focal_length(self):
        """ Get focal length
        :returns: Focal length as vector (fx, fy)
        """
        return self.focal_length



    def set_principal_point(self, principal_point):
        """ Set principal point
        The principal point is the intersection point of optical axis with chip
        and is defined in pixels coordinates (cx, cy)
        :param principal_point: Principal point
        """
        ppoint = np.asarray(principal_point)
        if ppoint.size != 2:
            raise ValueError('Provide 2d principal point')
        self.principal_point = ppoint



    def get_principal_point(self):
        """ Get principal point
        The principal point is the intersection point of optical axis with chip
        and is defined in pixels coordinates (cx, cy)
        :returns: Principal point
        """
        return self.principal_point



    def set_distortion(self, distortion):
        """ Set distortion parameters
        Parameters (k1, k2, p1, p2, k3) for radial (kx) and tangential (px) distortion
        :param distortion: Distortion parameters
        """
        dist = np.array(distortion)
        if dist.size != 5:
            raise ValueError('Provide 5d distortion vector')
        self.distortion = dist



    def get_distortion(self):
        """ Get distortion parameters
        Parameters (k1, k2, p1, p2, k3) for radial (kx) and tangential (px) distortion
        :returns: Distortion parameters
        """
        return self.distortion



    def set_camera_position(self, camera_position):
        """ Set camera position
        Transformation from world coordinate system to camera coordinate system as Trafo3d object
        :param camera_position: Camera position
        """
        self.camera_position = camera_position



    def get_camera_position(self):
        """ Get camera position
        Transformation from world coordinate system to camera coordinate system as Trafo3d object
        :returns: Camera position
        """
        return self.camera_position



    def json_save(self, filename):
        """ Save camera parameters to json file
        :param filename: Filename of json file
        """
        params = {}
        self.dict_save(params)
        with open(filename, 'w') as file_handle:
            json.dump(params, file_handle, indent=4, sort_keys=True)



    def dict_save(self, param_dict):
        """ Save camera parameters to dictionary
        :param params: Empty dictionary to store camera parameters in
        """
        param_dict['chip_size'] = self.chip_size.tolist()
        param_dict['focal_length'] = self.focal_length.tolist()
        param_dict['principal_point'] = self.principal_point.tolist()
        param_dict['distortion'] = self.distortion.tolist()
        param_dict['camera_position'] = {}
        param_dict['camera_position']['t'] = self.camera_position.get_translation().tolist()
        param_dict['camera_position']['q'] = self.camera_position.get_rotation_quaternion().tolist()



    def calculate_opening_angles(self):
        """ Calculate opening angles
        :returns: Opening angles in x and y in radians
        """
        p = np.array([[self.chip_size[1], self.chip_size[0], 1]])
        P = self.chip_to_scene(p)
        return 2.0 * np.arctan2(P[0, 0], P[0, 2]), \
            2.0 * np.arctan2(P[0, 1], P[0, 2])



    def scale_resolution(self, factor=1.0):
        """ Scale camera resolution
        The camera resolution heavily influences the computational resources needed
        to snap images. So for most setups it makes sense to keep a low resolution
        camera to take test images and then later to scale up the camera resolution.
        This method scales chip_size, f, c and distortion accordingly to increase
        (factor > 1) or reduce (factor > 1) camera resolution.
        :param factor: Scaling factor
        """
        self.chip_size = (factor * self.chip_size).astype(np.int64)
        self.focal_length = factor * self.focal_length
        self.principal_point = factor * self.principal_point
        self.distortion = factor * self.distortion



    def scene_to_chip(self, P):
        """ Transforms points in scene to points on chip
        This function does not do any clipping boundary checking!
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :returns: n points p=(u, v, d) on chip, shape (n, 3)
        """
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError('Provide scene coordinates of shape (n, 3)')
        # Transform points from world coordinate system to camera coordinate system
        P = self.camera_position.inverse() * P
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
        # focal length and principal point
        p = np.NaN * np.zeros(P.shape)
        p[valid, 0] = self.focal_length[0] * x3 + self.principal_point[0]
        p[valid, 1] = self.focal_length[1] * y3 + self.principal_point[1]
        p[valid, 2] = np.linalg.norm(P[valid, :], axis=1)
        return p



    def scene_points_to_depth_image(self, P, C=None):
        """ Transforms points in scene to depth image
        Image is initialized with np.NaN, invalid chip coordinates are filtered
        :param P: n points P=(X, Y, Z) in scene, shape (n, 3)
        :param C: n colors C=(R, G, B) for each point; same shape as P; optional
        :returns: Depth image, matrix of shape (self.chip_size[1], self.chip_size[0]),
            each element is distance; if C was provided, also returns color image
            of same size
        """
        p = self.scene_to_chip(P)
        # Clip image indices to valid points (can cope with NaN values in p)
        indices = np.round(p[:, 0:2]).astype(int)
        x_valid = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < self.chip_size[0])
        y_valid = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < self.chip_size[1])
        valid = np.logical_and(x_valid, y_valid)
        # Initialize empty image with NaN
        dImg = np.NaN * np.empty((self.chip_size[1], self.chip_size[0]))
        # Set image coordinates to distance values
        dImg[indices[valid, 1], indices[valid, 0]] = p[valid, 2]
        # If color values given, create color image as well
        if C is not None:
            if not np.array_equal(P.shape, C.shape):
                raise ValueError('P and C have to have the same shape')
            cImg = np.NaN * np.empty((self.chip_size[1], self.chip_size[0], 3))
            cImg[indices[valid, 1], indices[valid, 0], :] = C[valid, :]
            return dImg, cImg
        return dImg



    def chip_to_scene(self, p):
        """ Transforms points on chip to points in scene
        This function does not do any clipping boundary checking!
        :param p: n points p=(u, v, d) on chip, shape (n, 3)
        :returns: n points P=(X, Y, Z) in scene, shape (n, 3)
        """
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError('Provide chip coordinates of shape (n, 3)')
        # focal length and principal point
        x3 = (p[:, 0] - self.principal_point[0]) / self.focal_length[0]
        y3 = (p[:, 1] - self.principal_point[1]) / self.focal_length[1]
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
            -273*k1**5 + 364*k1**3*k2 - 78*k1*k2**2 - 78*k1**2*k3 +
            12*k2*k3 + 12*k1*k4,
            1428*k1**6 - 2380*k1**4*k2 + 840*k1**2*k2**2 - 35*k2**3 +
            560*k1**3*k3 -210*k1*k2*k3 + 7*k3**2 - 105*k1**2*k4 + 14*k2*k4,
            -7752*k1**7 + 15504*k1**5*k2 - 7752*k1**3*k2**2 +
            816*k1*k2**3 - 3876*k1**4*k3 + 2448*k1**2*k2*k3 - 136*k2**2*k3 -
            136*k1*k3**2 + 816*k1**3*k4 - 272*k1*k2*k4 + 16*k3*k4,
            43263*k1**8 - 100947*k1**6*k2 + 65835*k1**4*k2**2 -
            11970*k1**2*k2**3 + 285*k2**4 + 26334*k1**5*k3 -
            23940*k1**3*k2*k3 + 3420*k1*k2**2*k3 + 1710*k1**2*k3**2 -
            171*k2*k3**2 - 5985*k1**4*k4 + 3420*k1**2*k2*k4 - 171*k2**2*k4 -
            342*k1*k3*k4 + 9*k4**2,
            -246675*k1**9 + 657800*k1**7*k2 - 531300*k1**5*k2**2 +
            141680*k1**3*k2**3 - 8855*k1*k2**4 - 177100*k1**6*k3 +
            212520*k1**4*k2*k3 - 53130*k1**2*k2**2*k3 + 1540*k2**3*k3 -
            17710*k1**3*k3**2 + 4620*k1*k2*k3**2 - 70*k3**3 + 42504*k1**5*k4 -
            35420*k1**3*k2*k4 + 4620*k1*k2**2*k4 + 4620*k1**2*k3*k4 -
            420*k2*k3*k4 - 210*k1*k4**2
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
        P = self.camera_position * P
        return P



    def depth_image_to_scene_points(self, img):
        """ Transforms depth image to list of scene points
        :param img: Depth image, matrix of shape (self.chip_size[1], self.chip_size[0]),
            each element is distance or NaN
        :returns: n points P=(X, Y, Z) in scene, shape (n, 3) with
            n=np.prod(self.chip_size) - number of NaNs
        """
        if self.chip_size[0] != img.shape[1] or self.chip_size[1] != img.shape[0]:
            raise ValueError('Provide depth image of proper size')
        mask = ~np.isnan(img)
        if not np.all(img[mask] >= 0.0):
            raise ValueError('Depth image must contain only positive distances or NaN')
        x = np.arange(self.chip_size[0])
        y = np.arange(self.chip_size[1])
        Y, X = np.meshgrid(y, x, indexing='ij')
        p = np.vstack((X.flatten(), Y.flatten(), img.flatten())).T
        mask = np.logical_not(np.isnan(p[:, 2]))
        return self.chip_to_scene(p[mask])



    @staticmethod
    def __ray_mesh_intersect(rayorig, raydir, triangles):
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
    def __flat_shading(mesh, triangle_idx, lightvec):
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
    def __gouraud_shading(mesh, Pbary, triangle_idx, lightvec):
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
        rayorig = self.camera_position.get_translation()
        img = np.ones((self.chip_size[1], self.chip_size[0]))
        raydir = self.depth_image_to_scene_points(img) - rayorig
        # Do raytracing
        P = np.zeros(raydir.shape)
        Pbary = np.zeros(raydir.shape)
        triangle_idx = np.zeros(raydir.shape[0], dtype=int)
        for i in range(raydir.shape[0]):
            P[i, :], Pbary[i, :], triangle_idx[i] = \
                CameraModel.__ray_mesh_intersect(rayorig, raydir[i, :], mesh.triangle_vertices)
        # Reduce data to valid intersections of rays with triangles
        valid = ~np.isnan(P[:, 0])
        P = P[valid, :]
        Pbary = Pbary[valid, :]
        triangle_idx = triangle_idx[valid]
        # Calculate shading
        lightvec = -self.camera_position.get_rotation_matrix()[:, 2]
        if self.shading_mode == 'flat':
            C = CameraModel.__flat_shading(mesh, triangle_idx, lightvec)
        elif self.shading_mode == 'gouraud':
            C = CameraModel.__gouraud_shading(mesh, Pbary, triangle_idx, lightvec)
        # Determine color and depth images
        dImg, cImg = self.scene_points_to_depth_image(P, C)
        return dImg, cImg, P
