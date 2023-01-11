import matplotlib.pyplot as plt
plt.close('all')
import numpy as np
import os
import sys
import time
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, mesh_generate_plane, \
    mesh_generate_image_file, mesh_generate_charuco_board, \
    show_images, save_shot, load_shot



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
    show_images(depth_image, color_image)
    o3d.visualization.draw_geometries([cs, pcl])

    # Visualize camera rays; makes only sense with few pixels
    if False:
        rays = cam.get_camera_rays()
        rays.scale(300.0)
        rays_mesh = rays.get_mesh()
        o3d.visualization.draw_geometries([cs, plane, sphere, rays_mesh])

    # Save data
    cam.json_save('cam.json')
    save_shot('demo', depth_image, color_image, pcl)
