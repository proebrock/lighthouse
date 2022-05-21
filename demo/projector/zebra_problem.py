import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.shader_parallel_light import ShaderParallelLight
from camsimlib.shader_point_light import ShaderPointLight
from camsimlib.o3d_utils import mesh_generate_plane, show_images



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(60, 45),
                    )
    #cam.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam.scale_resolution(5)
    cam.place((-200, 0, 400))
    cam.look_at((-200, 0, 0))
    cam.roll(np.deg2rad(-90))

    # Object: Floor
    plane_floor = mesh_generate_plane((800, 800), color=(1, 1, 1))
    plane_floor.compute_triangle_normals()
    plane_floor.compute_vertex_normals()
    plane_floor.translate(-plane_floor.get_center())
    # Object: Wall
    plane_wall = mesh_generate_plane((200, 800), color=(1, 0, 0))
    plane_wall.compute_triangle_normals()
    plane_wall.compute_vertex_normals()
    plane_wall.translate(-plane_wall.get_center())
    T = Trafo3d(t=(0, 0, 100), rpy=np.deg2rad((0, -90, 0)))
    plane_wall.transform(T.get_homogeneous_matrix())
    # Object: Combine
    mesh = plane_floor + plane_wall

    # Shaders
    point_light = ShaderPointLight(light_position=(100, 0, 400))
    parallel_light = ShaderParallelLight(light_direction=(-1, 0, -1))

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        point_light_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        point_light_sphere.translate(point_light.get_light_position())
        point_light_sphere.paint_uniform_color((1, 1, 0))
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        o3d.visualization.draw_geometries([world_cs, point_light_sphere, mesh, \
            cam_cs, cam_frustum])

    # Snap image
    depth_image, color_image, pcl = cam.snap(mesh, [point_light])
    #depth_image, color_image, pcl = cam.snap(mesh, [parallel_light])

    # Visualize images
    show_images(depth_image, color_image)

    # Check results
    mask_nan = np.isnan(color_image)
    mask_shade = np.isclose(color_image, 0.0)
    lines_nan = np.sum(mask_nan) / 3 / cam.get_chip_size()[0]
    lines_shade = np.sum(mask_shade) / 3 / cam.get_chip_size()[0]
    print(lines_nan)
    print(lines_shade)
    #assert np.sum(mask_nan) / 3 == 67800 # 113 lines
    #assert np.sum(mask_shade) / 3 == 134400 # 224 lines
    #assert np.sum(np.logical_and(~mask_nan, ~mask_shade)) / 3 == 67800 # 113 lines
