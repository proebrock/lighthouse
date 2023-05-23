import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import time
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from common.image_utils import image_3float_to_rgb, image_float_to_rgb, image_show
from common.mesh_utils import mesh_generate_plane, mesh_transform
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel



if __name__ == '__main__':
    # Create camera and set its model parameters
    cam = CameraModel(chip_size=(40, 30),
                      focal_length=(50, 55),
                      distortion=(-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam.scale_resolution(8)
    cam.place((0, 0, 500))
    cam.look_at((10, 0, 0))
    cam.roll(np.deg2rad(90))
    cam.move_closer(50)

    # Setup scene: Plane
    plane = mesh_generate_plane((200, 200), color=(1, 1, 0))
#    plane = mesh_generate_image_file('../data/tux.png', pixel_size=3)
#    plane = mesh_generate_charuco_board((6, 5), 30.0)
    plane.translate(-plane.get_center())
    mesh_transform(plane, Trafo3d(rpy=np.deg2rad([-25, 25, 0])))
    # Setup scene: Sphere
    sphere = o3d.io.read_triangle_mesh('../../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere_radius = 50.0
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.translate((0, 0, -15))
    sphere.paint_uniform_color((1, 0, 0))

    # Visualize scene
    cs = cam.get_cs(size=50.0)
    frustum = cam.get_frustum(size=300.0)
    o3d.visualization.draw_geometries([cs, frustum, plane, sphere])

    # Snap image
    mesh = plane + sphere
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(mesh)
    toc = time.monotonic()
    print(f'Snapping image took {(toc - tic):.1f}s')

    # Visualize images and point cloud
    color_image = image_3float_to_rgb(color_image, nan_color=(0, 255, 255))
    image_show(color_image, 'Color image')
    depth_image = image_float_to_rgb(depth_image, cmap_name='viridis',
        min_max=None, nan_color=(0, 255, 255))
    image_show(depth_image, 'Depth image')
    plt.show()
    o3d.visualization.draw_geometries([cs, pcl])

    # Visualize camera rays; makes only sense with few pixels
    if False:
        rays = cam.get_rays()
        rays.scale(300.0)
        rays_mesh = rays.get_mesh()
        o3d.visualization.draw_geometries([cs, plane, sphere, rays_mesh])

