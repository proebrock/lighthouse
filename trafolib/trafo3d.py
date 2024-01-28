import copy
import numpy as np
import matplotlib.pyplot as plt
from . pyquaternion import Quaternion


class Trafo3d:

    def __init__(self, *args, **kwargs):
        """ Constructor

            Provide translational or rotatory initializers:

        't'    - Translational, see set_translation()
        'mat'  - Rotation matrix, see set_rotation_matrix()
        'hom'  - Homogeneous matrix, see set_homogeneous_matrix()
        'rpy'  - RPY angles, see set_rotation_rpy()
        'q'    - Quaternion, see set_rotation_quaternion()
        'rodr' - Rodrigues rotation formula (OpenCV), see set_rotation_rodrigues()
        'list' - Provide transformation as 7 element vector
                 (t and q, as provided by to_list())

        Do not provide multiple translational or multiple rotatory initializers.

        :param args: Non-keyworded fixed position arguments
        :param kwargs: Keyworded arguments
        """
        if len(args) > 0:
            raise ValueError('No positional arguments allowed')
        if len(frozenset(kwargs.keys()).intersection(set(('t', 'hom')))) >= 2:
            raise ValueError('Multiple translational components defined')
        if len(frozenset(kwargs.keys()).intersection(set(('mat', 'hom', 'rpy', 'q', 'rodr')))) >= 2:
            raise ValueError('Multiple rotational components defined')
        if not frozenset(kwargs.keys()).issubset(set(('t', 'mat', 'hom', 'rpy',
                                                      'q', 'rodr', 'list'))):
            raise ValueError('Unknown arguments: ' + str(kwargs.keys()))
        self._t = np.zeros(3)
        self._r = Quaternion()
        if 't' in kwargs:
            self.set_translation(kwargs['t'])
        if 'mat' in kwargs:
            self.set_rotation_matrix(kwargs['mat'])
        if 'hom' in kwargs:
            self.set_homogeneous_matrix(kwargs['hom'])
        if 'rpy' in kwargs:
            self.set_rotation_rpy(kwargs['rpy'])
        if 'q' in kwargs:
            self.set_rotation_quaternion(kwargs['q'])
        if 'rodr' in kwargs:
            self.set_rotation_rodrigues(kwargs['rodr'])
        if 'list' in kwargs:
            if len(kwargs) > 1:
                raise ValueError('If specifying "list", specify no other elements')
            if len(kwargs['list']) != 7:
                raise ValueError('If specifying "list", length has to be 7')
            self.set_translation(kwargs['list'][0:3])
            self.set_rotation_quaternion(kwargs['list'][3:])


    def __str__(self):
        """ Get readable string representation of object
        :return: String representation of object
        """
        tstr = np.array2string(self._t, precision=1, separator=', ', suppress_small=True)
        rpy = np.rad2deg(self.get_rotation_rpy())
        rstr = np.array2string(rpy, precision=1, separator=', ', suppress_small=True)
        return '(' + tstr + ', ' + rstr + ')'


    def __repr__(self):
        """ Get unambiguous string representation of object
        :return: String representation of object
        """
        return repr(self._t) + ',' + repr(self._r.elements)


    def __eq__(self, other):
        """ Check transformations for equality
        :param other: Translation to compare with
        :return: True if transformations are equal, False otherwise
        """
        if isinstance(other, self.__class__):
            return np.allclose(self._t, other._t) and \
                (np.allclose(self._r.elements, other._r.elements) or \
                np.allclose(self._r.elements, -1.0 * other._r.elements))
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
        return self.__class__(t=self._t, q=self._r.elements)


    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(t=copy.deepcopy(self._t, memo), q=copy.deepcopy(self._r.elements, memo))
        memo[id(self)] = result
        return result


    def to_list(self):
        """ Provides transformation as list of values
        Usage is for serialization; deserialize with constructor parameter 'list'
        :return: Transformation as list of values
        """
        return [self._t[0], self._t[1], self._t[2], \
            self._r.elements[0], self._r.elements[1], \
            self._r.elements[2], self._r.elements[3]]


    def dict_save(self, param_dict):
        """ Save object to dictionary
        :param param_dict: Dictionary to store data in
        """
        param_dict['t'] = self.get_translation().tolist()
        param_dict['q'] = self.get_rotation_quaternion().tolist()


    def dict_load(self, param_dict):
        """ Load object from dictionary
        :param param_dict: Dictionary with data
        """
        self.set_translation(param_dict['t'])
        self.set_rotation_quaternion(param_dict['q'])


    def plot2d(self, ax, normal=2, scale=1.0, label=None):
        """ Plotting the tranformation as coordinate system projected into a 2D plane
        :param ax: Axes object created by fig = plt.figure(); ax = fig.add_subplot(111) or similar
        :param normal: Normal vector of plane 0->X-vector->YZ-plane,
                       1->Y-vector->XZ-plane, 2->Z-vector->XY-plane
        :param scale: Scale factor for axis lengths
        :param label: Label printed close to coordinate system
        """
        plane = np.arange(3)
        if not normal in plane:
            raise ValueError('Unknown axis: ' + str(normal))
        plane = plane[plane != normal]
        t = self.get_translation()
        origin = t[plane]
        m = self.get_rotation_matrix()
        u = m[plane, :]
        colors = ('r', 'g', 'b')
        for i in range(3):
            ui = u[:, i]
            if np.linalg.norm(ui) > 0.1:
                ax.annotate('', xy=origin+ui*scale, xytext=origin,
                    arrowprops=dict(arrowstyle="->", color=colors[i]))
            else:
                c = plt.Circle(origin, 0.15*scale, ec=colors[i], fc='w')
                ax.add_artist(c)
                sign = np.array([1, -1, 1])
                marker = 'o' if (m[normal, i] * sign[normal]) > 0 else 'x'
                ax.plot(*origin, marker=marker, color=colors[i], markersize=5*scale)
        if label is not None:
            ax.text(*(origin+0.2*scale), label, color='k')


    def plot_axis_angle(self, ax, scale=1.0, label=None):
        """ Plotting the transformation as rotation axis in 3d
        The axis has a length of 1.0 and is plotted in black; a label denotes the angle in deg
        :param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
        :param scale: Scale factor for axis lengths
        :param label: In None, no label shown, if True angle in deg shown, else given text shown
        """
        origin = self.get_translation()
        axis = self._r.axis * scale
        angle = np.rad2deg(self._r.angle)
        ax.quiver3D(*origin, *axis, color='k', arrow_length_ratio=0.15)
        if label is not None:
            if isinstance(label, bool):
                if label:
                    l = f'${angle:.1f}^\\circ$'
                else:
                    label = ''
            else:
                l = label
            ax.text(*(origin+axis+0.1*scale), l, color='k',
                    verticalalignment='center', horizontalalignment='center')


    def plot_rodrigues(self, ax, scale=1.0, label=None):
        """ Plotting the transformation as Rodrigues vector in 3d
        The direction and length is determined by the Rodrigues vector
        :param ax: Axes object created by fig = plt.figure(); ax = fig.gca(projection='3d')
        :param scale: Scale factor for axis lengths
        :param label: In None, no label shown, if True angle in deg shown, else given text shown
        """
        origin = self.get_translation()
        axis = self.get_rotation_rodrigues() * scale
        ax.quiver3D(*origin, *axis, color='k', arrow_length_ratio=0.15)
        if label is not None:
            if isinstance(label, bool):
                if label:
                    l = f'${np.rad2deg(self._r.angle):.1f}^\\circ$'
                else:
                    label = ''
            else:
                l = label
            ax.text(*(origin+axis+0.1*scale), l, color='k',
                    verticalalignment='center', horizontalalignment='center')


    def plot3d(self, ax, scale=1.0, label=None):
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


    def plot_frustum3d(self, ax, scale=1.0, label=None):
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
            [0, 0, 0],
            [-e, -e, z],
            [0, 0, 0],
            [-e, e, z],
            [0, 0, 0],
            [e, e, z],
            [0, 0, 0],
            [e, -e, z],
            [-e, -e, z],
            [-e, e, z],
            [-e, e, z],
            [e, e, z],
            [e, e, z],
            [e, -e, z],
            [e, -e, z],
            [-e, -e, z]
            ])
        points = self * points
        for i in range(0, points.shape[0], 2):
            ax.plot(points[i:i+2, 0], points[i:i+2, 1],
                    points[i:i+2, 2], color='b')
        # Label
        if label is not None:
            ax.text(*(origin+0.05), label, color='k')


    def set_translation(self, value):
        """ Set translatory component of transformation
        :param value: Translation as vector (x, y, z)
        """
        value = np.asarray(value)
        if value.size != 3:
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        self._t = np.reshape(value, (3,))


    def get_translation(self):
        """ Get translatory component of transformation
        :return: Translation as vector (x, y, z)
        """
        return self._t


    def set_rotation_matrix(self, value):
        """ Set rotatory component of transformation as rotation matrix
        :param value: 3x3 rotation matrix
        """
        value = np.asarray(value)
        if value.shape != (3, 3):
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        self._r = Quaternion(matrix=value)


    def get_rotation_matrix(self):
        """ Get rotatory component of transformation as rotation matrix
        :return: 3x3 rotation matrix
        """
        return self._r.rotation_matrix


    def set_homogeneous_matrix(self, value):
        """ Set translation as homogenous matrix
        :param value: 4x4 homogenous matrix
        """
        value = np.asarray(value)
        if value.shape != (4, 4):
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        self.set_translation(value[0:3, 3])
        self.set_rotation_matrix(value[0:3, 0:3])


    def get_homogeneous_matrix(self):
        """ Get translation as homogenous matrix
        :return: 4x4 homogenous matrix
        """
        result = np.identity(4)
        result[0:3, 3] = self.get_translation().reshape((3,))
        result[0:3, 0:3] = self.get_rotation_matrix()
        return result


    def set_rotation_rpy(self, value):
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
            [cz * cy, -sz * cx + cz * sy * sx, sz * sx + cz * sy * cx],
            [sz * cy, cz * cx + sz * sy * sx, -cz * sx + sz * sy * cx],
            [-sy, cy * sx, cy * cx]
            ])
        self._r = Quaternion(matrix=m)


    @staticmethod
    def wrap_angles(angles):
        """ Wrap any input angle(s) to the range (-pi..pi]
        :param angle: Input angle(s)
        :return: Wrapped angle(s)
        """
        if isinstance(angles, (list, np.ndarray)):
            result = np.mod(np.asarray(angles) + np.pi, 2.0 * np.pi) - np.pi
            result[result == -np.pi] = np.pi
        else:
            result = ((angles + np.pi) % (2.0 * np.pi)) - np.pi
            if result == -np.pi:
                result = np.pi
        return result


    def get_rotation_rpy(self):
        """ Get rotatory component of transformation as roll-pitch-yaw angles
        :return: RPY angles in radians as vector (x_roll, y_pitch, z_yaw)
        """
        m = self._r.rotation_matrix
        z = np.arctan2(m[1, 0], m[0, 0])
        y = np.arctan2(-m[2, 0], m[0, 0] * np.cos(z) + m[1, 0] * np.sin(z))
        x = np.arctan2(m[0, 2] * np.sin(z) - m[1, 2] * np.cos(z),
                       -m[0, 1] * np.sin(z) + m[1, 1] * np.cos(z))
        result = np.array([x, y, z])
        result = Trafo3d.wrap_angles(result)
        return result


    def set_rotation_quaternion(self, value):
        """ Set rotatory component of transformation as unit quarternion
        :param value: Unit quarternion (w, x, y, z)
        """
        value = np.asarray(value)
        if value.shape != (4,):
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        if not np.isclose(np.linalg.norm(value), 1.0):
            raise ValueError('Unit quaternion expected')
        self._r = Quaternion(value)


    def get_rotation_quaternion(self):
        """ Get rotatory component of transformation as unit quarternion
        :return: Unit quarternion (w, x, y, z)
        """
        return self._r.elements


    def set_rotation_rodrigues(self, value):
        """ Set rotatory component of transformation as Rodrigues rotation formula (OpenCV)
        :param value: Rodrigues rotation formula (x, y, z)
        """
        value = np.asarray(value)
        if value.size != 3:
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        value = np.reshape(value, (3,))
        theta = np.linalg.norm(value)
        if np.isclose(theta, 0.0):
            self._r = Quaternion()
        else:
            axis = value / theta
            self._r = Quaternion(axis=axis, radians=theta)


    def get_rotation_rodrigues(self):
        """ Get rotatory component of transformation as Rodrigues rotation formula (OpenCV)
        :return: Rodrigues rotation formula (x, y, z)
        """
        return self._r.axis * self._r.angle


    def inverse(self):
        """ Get inverse transformation as a new object
        :return: Inverse transformation
        """
        r = self._r.inverse
        t = -1.0 * r.rotate(self._t)
        result = self.__class__()
        result.set_translation(t)
        result.set_rotation_quaternion(r.elements)
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
            result._t = self._t.reshape((3,)) + self._r.rotate(other._t).reshape((3,))
            result._r = self._r * other._r
            return result
        if isinstance(other, (list, np.ndarray)):
            other = np.asarray(other)
            if other.ndim == 1 and other.size == 3:
                other = np.reshape(other, (3, 1))
                return np.reshape(self._t, (3,)) + self._r.rotate(other.ravel())
            if other.ndim == 2 and other.shape[1] == 3:
                t = np.tile(self._t, (other.shape[0], 1))
                r = np.dot(other, self._r.rotation_matrix.T)
                return t + r
            raise ValueError('Expecting dimensions (3,) or (n,3)')
        raise ValueError('Expecting instance of Trafo3d or numpy array')


    def interpolate(self, other, weight=0.5):
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
        result._t = (1.0 - weight) * self._t + weight * other._t
        result._r = Quaternion.slerp(self._r, other._r, weight)
        return result


    @staticmethod
    def average(trafos, weights=None):
        """ Calculate average of a number of transformations
        :param trafos: List of Trafo3d objects
        :param weights: Individual weights for each transformation (optional)
        If no weights specified, all trafos are weighted equally.
        Number of trafos and weights must be the same.
        The sum(weights) cannot be zero.
        """
        if len(trafos) == 0:
            return None
        if weights is None:
            ws = np.ones(len(trafos))
        else:
            ws = np.asarray(weights)
            if len(trafos) != ws.size:
                raise ValueError('Provide same number of trafos and weights')
            if np.isclose(np.sum(ws), 0.0):
                raise ValueError('Weight sum cannot be zero')
        # Average translations
        t = np.zeros(3)
        for trafo, w in zip(trafos, ws):
            t += w * trafo.get_translation()
        t = t / np.sum(ws)
        # Average rotations
        # Original:
        # Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
        # "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
        # no. 4 (2007): 1193-1197.
        # Matlab code:
        # https://github.com/tolgabirdal/averaging_quaternions/blob/master/wavg_quaternion_markley.m
        A = np.zeros((4, 4))
        for trafo, w in zip(trafos, ws):
            q = trafo.get_rotation_quaternion()
            A = A + np.outer(q, q) * w
        A = A /  np.sum(ws)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        max_eigenvalue_index = np.argsort(eigenvalues)[-1]
        q = np.real(eigenvectors[:, max_eigenvalue_index])
        # Combine translation and rotation to result
        return Trafo3d(t=t, q=q)


    def distance(self, other):
        """ Calculate difference between two transformations:
        For translational component - Euclidian distance
        For rotatory component - Absolute of rotation angle around rotation axis (quaternion)
        """
        delta = self.inverse() * other
        dt = np.linalg.norm(delta._t)
        dr = np.abs(delta._r.angle)
        #dt = np.linalg.norm(self._t - other._t)
        #dr = Quaternion.absolute_distance(self._r, other._r)
        return dt, dr


    @staticmethod
    def average_and_errors(trafos):
        """Interpolate an array of transformations and return average and errors
        Average uses Slerps (spherical linear interpolation) for rotatory component
        Each row of error matrix contains Trafo3d.Distance of average and trafo
        Error matrix size: len(trafos) x 2
        """
        if len(trafos) == 0:
            return None, None
        average = Trafo3d.average(trafos)
        errors = np.zeros((len(trafos), 2))
        for i, t in enumerate(trafos):
            dt, dr = average.distance(t)
            errors[i, :] = [dt, dr]
        return average, errors


    def get_principal_plane(self, plane_str='xy'):
        """ Get equation of principal plane (XY, YZ or XZ plane)
        Plane equation: All points (x,y,z) are on plane that fulfill the
        equation nx*x + ny*y + nz*z + d = 0 with sqrt(nx**2 + ny**2 + nz**2) == 1
        :param: String determining plane: 'xy', 'yz' or 'xz'
        :return: Plane, shape (4, ), see above
        """
        plane_index = [ 'yz', 'xz', 'xy' ].index(plane_str)
        R = self.get_rotation_matrix()
        n = R[:, plane_index]
        d = -np.sum(self._t * n)
        return np.array((*n, d))

