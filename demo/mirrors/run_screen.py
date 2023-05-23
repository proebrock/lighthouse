import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from scipy.stats import multivariate_normal

sys.path.append(os.path.abspath('../../'))
from common.mesh_utils import mesh_generate_surface
from trafolib.trafo3d import Trafo3d
from camsimlib.multi_mesh import MultiMesh
from camsimlib.camera_model import CameraModel
from camsimlib.screen import Screen



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(100, 100),
                    )
    cam.scale_resolution(20)
    cam.place((200, -200, 200))
    cam.look_at((200, 500, 0))
    cam.roll(np.deg2rad(90))
    cam_cs = cam.get_cs(50.0)
    cam_frustum = cam.get_frustum(200.0)

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
    screen_pose = Trafo3d(t=(0.0, 350.0, 360.0), rpy=np.deg2rad((-60, 0, 0)))
    width_mm = 400.0
    height_mm = (image.shape[0] * width_mm) / image.shape[1]
    screen = Screen((width_mm, height_mm), image, screen_pose)
    screen_mesh = screen.get_mesh()

    # Generate object
    fun = lambda x: multivariate_normal.pdf(x, cov=[[0.2, 0.0], [0.0, 0.2]])
    surface = mesh_generate_surface(fun, xrange=(-10.0, 10.0), yrange=(-10.0, 10.0),
        num=(100, 100), scale=(400.0, 400.0, -1.5))

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        cam_cs = cam.get_cs(50.0)
        cam_frustum = cam.get_frustum(200.0)
        screen_cs = screen.get_cs(50.0)
        o3d.visualization.draw_geometries([world_cs, cam_cs, cam_frustum,
            screen_cs, screen_mesh, surface])

    # Snap image
    meshes = MultiMesh([screen_mesh, surface], [False, True])
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(meshes)
    toc = time.monotonic()
    print(f'Snapping image took {(toc - tic):.1f}s')

    # Show image
    fig, ax = plt.subplots()
    nanidx = np.where(np.isnan(color_image))
    img = color_image.copy()
    img[nanidx[0], nanidx[1], :] = (0, 1, 1)
    ax.imshow(img)
    ax.set_axis_off()
    plt.show()
