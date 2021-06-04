# -*- coding: utf-8 -*-
""" Simulation of a random dot projector
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d



class DotProjector:

    def __init__(self, projector_pose=None):
        # Point configuration
        self.chip_size = (9, 7)
        self.chip_xlim = np.array((-0.7, 0.7))
        self.chip_ylim = np.array((-0.5, 0.5))
        # Distances between points in x and y
        dx = (self.chip_xlim[1] - self.chip_xlim[0]) / (self.chip_size[0] + 2)
        dy = (self.chip_ylim[1] - self.chip_ylim[0]) / (self.chip_size[1] + 2)
        # Arrange points in a grid
        self.x = np.linspace(self.chip_xlim[0] + dx,
                             self.chip_xlim[1] - dx,
                             self.chip_size[0])
        self.y = np.linspace(self.chip_ylim[0] + dy,
                             self.chip_ylim[1] - dy,
                             self.chip_size[1])
        self.chip_points = np.empty((self.chip_size[0] * self.chip_size[1], 3))
        self.chip_points[:,0:2] = np.stack(np.meshgrid(self.x, self.y), -1).reshape(-1,2)
        self.chip_points[:,2] = 1.0
        # Randomly move points around
        noise_gain = 0.9
        dx_max = (noise_gain * dx) / 2
        dy_max = (noise_gain * dy) / 2
        self.chip_points[:,0] = self.chip_points[:, 0] + \
            np.random.uniform(low=-dx_max, high=dx_max, size=self.chip_points.shape[0])
        self.chip_points[:,1] = self.chip_points[:, 1] + \
            np.random.uniform(low=-dy_max, high=dy_max, size=self.chip_points.shape[0])
        # Projector position: transformation from world to projector
        self.projector_pose = None
        if projector_pose is None:
            self.set_projector_pose(Trafo3d())
        else:
            self.set_projector_pose(projector_pose)



    def set_projector_pose(self, projector_pose):
        """ Set projector pose
        Transformation from world coordinate system to projector coordinate system as Trafo3d object
        :param projector_pose: Projector position
        """
        self.projector_pose = projector_pose



    def get_rays(self):
        """ Get rays emitted by dot projector
        :return: Ray origin (origin of projector pose) and ray directions as normalized vectors
        """
        # Ray origin
        rayorig = self.projector_pose.get_translation()
        # Normalized ray directions
        chip_points_in_world = self.projector_pose * self.chip_points
        raydir = chip_points_in_world - rayorig
        raydir = raydir / np.linalg.norm(raydir, axis=1)[:,np.newaxis]
        return rayorig, raydir



    def intersect_plane(self, plane_pose):
        """ Intersect projector rays with a plane
        :param plane_pose: Plane given as pose; X/Y-plane is plane
        :return: Intersection points
        """
        rayorig, raydir = self.get_rays()
        # Plane definition: n is a normal vector to the plane and p0 is a point on the plane
        plane_n = plane_pose.get_rotation_matrix()[:,2]
        plane_p0 = plane_pose.get_translation()
        # Calculate numerator and denominator for solution and check for denom==0
        numerator = np.dot(plane_p0 - rayorig, plane_n)
        denominator = np.dot(raydir, plane_n)
        mask_valid = ~np.isclose(denominator, 0)
        # Calculate fraction for valids only
        d = -np.ones_like(denominator)
        d[mask_valid] = numerator / denominator[mask_valid]
        # # Consider only solutions in positive direction of ray
        mask_positive = d >= 0
        return (raydir[mask_positive,:] * d[mask_positive, np.newaxis]) + rayorig



    def plot_chip(self):
        """ Plot view of projector chip and its points
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.chip_points[:,0], self.chip_points[:,1], 'ob')
        ax.set_xlim(self.chip_xlim)
        ax.set_xticks(self.x)
        ax.set_ylim(self.chip_ylim)
        ax.set_yticks(self.y)
        ax.set_title(f'Projector chip view of {self.chip_size[0]}x{self.chip_size[1]}={np.prod(self.chip_size)} points')
        ax.grid()
        plt.show()



    def get_cs(self, size=1.0):
        """ Returns a representation of the coordinate system of the projector
        Returns Open3d TriangleMesh object representing the coordinate system
        of the projector that can be used for visualization
        :param size: Length of the coordinate axes of the coordinate system
        :return: Coordinate system as Open3d mesh object
        """
        coordinate_system = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        coordinate_system.transform(self.projector_pose.get_homogeneous_matrix())
        return coordinate_system



    def get_frustum(self, size=1.0, color=(0, 0, 0)):
        """ Returns a representation of the frustum of the projector
        Returns Open3d LineSet object representing the frustum
        of the projector that can be used for visualization.
        (A "frustum" is a cone with the top cut off.)
        :param size: Length of the sides of the frustum
        :param color: Color of the frustum
        :return: Frustum as Open3d mesh object
        """
        P = np.array([
                [ self.chip_xlim[0], self.chip_ylim[0], 1.0 ],
                [ self.chip_xlim[0], self.chip_ylim[1], 1.0 ],
                [ self.chip_xlim[1], self.chip_ylim[0], 1.0 ],
                [ self.chip_xlim[1], self.chip_ylim[1], 1.0 ]
                ])
        P = P / np.linalg.norm(P, axis=1)[:,np.newaxis]
        P = P * size
        P = self.projector_pose * P
        # Create line set
        line_set = o3d.geometry.LineSet()
        points = np.vstack((self.projector_pose.get_translation(), P))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(color, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set


