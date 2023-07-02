# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from trafolib.trafo3d import Trafo3d
from common.mesh_utils import mesh_generate_plane
from . camera_model import CameraModel
from . shader_parallel_light import ShaderParallelLight
from . shader_point_light import ShaderPointLight



def test_illuminated_points():
    # To find the set of illuminated points (or shading points), we use the
    # intersection points P of the first raytracer run and start a secondary
    # raytracing from P towards the light source. If this secondary
    # raytracing intersects with the mesh between P and the light source, the
    # point is not illuminated by the light source.
    # Unfortunately P is located on the mesh, sometimes some epsilon above or
    # below the mesh; this means that the secondary raytracing may yield points
    # very close to P, so the detection of illuminated points is almost random.
    # There are two ways to solve this:
    #
    # * Filter the raytracer results for intersections very close to the origin
    #   of the ray ("scale" close to zero); in the Python implementation this is
    #   simple, but in the Embree implementation we have limited access to the
    #   API via Open3D
    # * Move P slightly above the mesh using the triangle normal vectors; it is
    #   a bit of a hack but seems to work
    #
    # This test case checks if the shadow point detection works properly

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
    plane_floor.translate(-plane_floor.get_center())
    # Object: Wall
    plane_wall = mesh_generate_plane((200, 800), color=(1, 0, 0))
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
