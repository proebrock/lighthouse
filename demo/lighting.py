import matplotlib.pyplot as plt
plt.close('all')
import numpy as np
import os
import sys
import time
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, show_images, save_shot


if __name__ == '__main__':
    # Create camera and set its model parameters
    cam = CameraModel(chip_size=(40, 30),
                      focal_length=(50, 55),
                      distortion=(-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam.scale_resolution(2)
    cam.set_lighting_mode('point')
    cam.set_light_vector((100, 0, 0))

    mesh = o3d.io.read_triangle_mesh('../data/knot.ply')
    mesh.paint_uniform_color((1, 0, 0))
    mesh.translate(-mesh.get_center()) # De-mean
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.scale(1, center=mesh.get_center())
    mesh_transform(mesh, Trafo3d(t=[0, 0, 350], rpy=np.deg2rad([-5, 10, 0])))

    if False:
        cs = cam.get_cs(size=50.0)
        frustum = cam.get_frustum(size=300.0)
        o3d.visualization.draw_geometries([cs, frustum, mesh])

    if True:
        tic = time.monotonic()
        depth_image, color_image, pcl = cam.snap(mesh)
        toc = time.monotonic()
        print(f'Snapping image took {(toc - tic):.1f}s')
        #show_images(depth_image, color_image)