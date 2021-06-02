# -*- coding: utf-8 -*-
""" Simulation of a random dot projector
"""

import matplotlib.pyplot as plt
import numpy as np
from trafolib.trafo3d import Trafo3d



class DotProjector:

    def __init__(self, projector_pose=None):
        # Point configuration
        self.chip_size = (9, 7)
        self.chip_xlim = np.array((-1, 1))
        self.chip_ylim = np.array((-1, 1))
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
        self.chip_points = np.stack(np.meshgrid(self.x, self.y), -1).reshape(-1,2)
        # Move points around
        noise_gain = 0.9
        dx_max = (noise_gain * dx) / 2
        dy_max = (noise_gain * dy) / 2
        self.chip_points[:,0] = self.chip_points[:, 0] + \
            np.random.uniform(low=-dx_max, high=dx_max, size=self.chip_points.shape[0])
        self.chip_points[:,1] = self.chip_points[:, 1] + \
            np.random.uniform(low=-dy_max, high=dy_max, size=self.chip_points.shape[0])
        # projector position: transformation from world to projector
        self.projector_pose = None
        if projector_pose is None:
            self.set_projector_pose(Trafo3d())
        else:
            self.set_projector_pose(projector_pose)



    def set_projector_pose(self, projector_pose):
        """ Set camera position
        Transformation from world coordinate system to projector coordinate system as Trafo3d object
        :param projector_pose: Projector position
        """
        self.projector_pose = projector_pose



    def plot_chip(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.chip_points[:,0], self.chip_points[:,1], 'ob')
        ax.set_xlim(self.chip_xlim)
        ax.set_xticks(self.x)
        ax.set_ylim(self.chip_ylim)
        ax.set_yticks(self.y)
        ax.set_title(f'Chip view of {self.chip_size[0]}x{self.chip_size[1]}={np.prod(self.chip_size)} points')
        ax.grid()
