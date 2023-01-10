import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from scipy.stats import multivariate_normal

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.screen import Screen
from camsimlib.o3d_utils import mesh_generate_surface



if __name__ == '__main__':
    # Create image of horizontal stripes
    image = np.zeros((600, 800, 3), dtype=np.uint8)
    stripe_width = 20
    for i in range(0, image.shape[0]//stripe_width, 2):
        start_index = i * stripe_width
        end_index = (i + 1) * stripe_width
        image[start_index:end_index, :, :] = 255
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()

    # Put image on screen
    screen_pose = Trafo3d(t=(100.0, 0, 0), rpy=np.deg2rad((0, 0, 0)))
    width_mm = 200.0
    height_mm = (image.shape[0] * width_mm) / image.shape[1]
    screen = Screen((width_mm, height_mm), image, screen_pose)

    # Generate object
    fun = lambda x: multivariate_normal.pdf(x, cov=[[0.5, 0.0], [0.0, 0.5]])
    object = mesh_generate_surface(fun, xrange=(-10.0, 10.0), yrange=(-10.0, 10.0),
        num=(4, 3), scale=(100.0, 100.0))

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        screen_cs = screen.get_cs(100.0)
        screen_viz = screen.get_mesh()
        o3d.visualization.draw_geometries([world_cs, screen_cs, screen_viz])
