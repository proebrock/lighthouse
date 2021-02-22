# -*- coding: utf-8 -*-


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
        self.t = np.zeros(3)
        self.r = Quaternion()
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
        tstr = np.array2string(self.t, precision=1, separator=', ', suppress_small=True)
        rpy = np.rad2deg(self.get_rotation_rpy())
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
        return self.__class__(t=self.t, q=self.r.elements)


    def __deepcopy__(self, memo):
        """ Deep copy
        :param memo: Memo dictionary
        :return: A deep copy of self
        """
        result = self.__class__(t=copy.deepcopy(self.t, memo), q=copy.deepcopy(self.r.elements, memo))
        memo[id(self)] = result
        return result


    def to_list(self):
        """ Provides transformation as list of values
        Usage is for serialization; deserialize with constructor parameter 'list'
        :return: Transformation as list of values
        """
        return [self.t[0], self.t[1], self.t[2], \
            self.r.elements[0], self.r.elements[1], \
            self.r.elements[2], self.r.elements[3]]


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
                ax.quiver(*origin, *(ui * scale), color=colors[i],
                          angles='xy', scale_units='xy', scale=1.0)
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
        axis = self.r.axis * scale
        angle = np.rad2deg(self.r.angle)
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
                    l = f'${np.rad2deg(self.r.angle):.1f}^\\circ$'
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
        self.t = np.reshape(value, (3,))


    def get_translation(self):
        """ Get translatory component of transformation
        :return: Translation as vector (x, y, z)
        """
        return self.t


    def set_rotation_matrix(self, value):
        """ Set rotatory component of transformation as rotation matrix
        :param value: 3x3 rotation matrix
        """
        value = np.asarray(value)
        if value.shape != (3, 3):
            raise ValueError('Initialization with invalid shape: ', str(value.shape))
        self.r = Quaternion(matrix=value)


    def get_rotation_matrix(self):
        """ Get rotatory component of transformation as rotation matrix
        :return: 3x3 rotation matrix
        """
        return self.r.rotation_matrix


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
        self.r = Quaternion(matrix=m)


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
        m = self.r.rotation_matrix
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
        self.r = Quaternion(value)


    def get_rotation_quaternion(self):
        """ Get rotatory component of transformation as unit quarternion
        :return: Unit quarternion (w, x, y, z)
        """
        return self.r.elements


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
            self.r = Quaternion()
        else:
            axis = value / theta
            self.r = Quaternion(axis=axis, radians=theta)


    def get_rotation_rodrigues(self):
        """ Get rotatory component of transformation as Rodrigues rotation formula (OpenCV)
        :return: Rodrigues rotation formula (x, y, z)
        """
        return self.r.axis * self.r.angle


    def inverse(self):
        """ Get inverse transformation as a new object
        :return: Inverse transformation
        """
        r = self.r.inverse
        t = -1.0 * r.rotate(self.t)
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
            result.t = self.t.reshape((3,)) + self.r.rotate(other.t).reshape((3,))
            result.r = self.r * other.r
            return result
        if isinstance(other, (list, np.ndarray)):
            other = np.asarray(other)
            if other.ndim == 1 and other.size == 3:
                other = np.reshape(other, (3, 1))
                return np.reshape(self.t, (3,)) + self.r.rotate(other)
            if other.ndim == 2 and other.shape[1] == 3:
                t = np.tile(self.t, (other.shape[0], 1))
                r = np.dot(other, self.r.rotation_matrix.T)
                return t + r
            raise ValueError('Expecting dimensions (3,) or (n,3)')
        raise ValueError('Expecting instance of Trafo3d or numpy array')


    @staticmethod
    def quaternion_slerp(q0, q1, weight):
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


    def interpolate(self, other, weight=0.5, shortest_path=True):
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
            result.r = Trafo3d.quaternion_slerp(self.r, other.r, weight)
        else:
            result.r = Quaternion.slerp(self.r, other.r, weight)
        return result


    @staticmethod
    def interpolate_multiple(trafos, shortest_path=True):
        """ Interpolate an array of transformations
        Uses Slerps (spherical linear interpolation) for rotatory component
        :param trafos: List of transformations
        :return: Resulting interpolated transformation
        """
        if len(trafos) == 0:
            return None
        result = trafos[0]
        for i, t in enumerate(trafos[1:]):
            result = result.interpolate(t, weight=1.0/(i+2), shortest_path=shortest_path)
        return result


    def distance(self, other):
        """ Calculate difference between two transformations:
        For translational component - Euclidian distance
        For rotatory component - Absolute of rotation angle around rotation axis (quaternion)
        """
        delta = self.inverse() * other
        dt = np.linalg.norm(delta.t)
        dr = np.abs(delta.r.angle)
        #dt = np.linalg.norm(self.t - other.t)
        #dr = Quaternion.absolute_distance(self.r, other.r)
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
        average = Trafo3d.interpolate_multiple(trafos, True)
        errors = np.zeros((len(trafos), 2))
        for i, t in enumerate(trafos):
            dt, dr = average.distance(t)
            errors[i, :] = [dt, dr]
        return average, errors
