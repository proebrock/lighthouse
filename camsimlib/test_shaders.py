# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import random as rand

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from trafolib.trafo3d import Trafo3d
from . camera_model import CameraModel
from . shader_parallel_light import ShaderParallelLight
from . shader_point_light import ShaderPointLight
from . o3d_utils import mesh_generate_plane, show_images



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def test_illuminated_points():
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(240, 180),
                    )
    cam.scale_resolution(5)
    cam.place((-100, 0, 200))
    cam.look_at((-100, 0, 0))
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

    # Shaders: Both result in same shadow
    point_light = ShaderPointLight(light_position=(200, 0, 400))
    parallel_light = ShaderParallelLight(light_direction=(-1, 0, -1))

    # Visualize scene
    if False:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        point_light_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        point_light_sphere.translate(point_light.get_light_position())
        point_light_sphere.paint_uniform_color((1, 1, 0))
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        o3d.visualization.draw_geometries([world_cs, point_light_sphere, mesh, \
            cam_cs, cam_frustum])

    # Snap images and check results
    depth_image, color_image, _ = cam.snap(mesh, [point_light])
    #show_images(depth_image, color_image)
    assert np.all(np.isclose(color_image, 0.0))

    _, color_image, _ = cam.snap(mesh, [parallel_light])
    #show_images(depth_image, color_image)
    assert np.all(np.isclose(color_image, 0.0))



if __name__ == '__main__':
    pytest.main()
