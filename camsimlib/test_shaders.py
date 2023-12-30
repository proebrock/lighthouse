import pytest

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from trafolib.trafo3d import Trafo3d
from common.mesh_utils import mesh_generate_plane
from . camera_model import CameraModel
from . shader_parallel_light import ShaderParallelLight
from . shader_point_light import ShaderPointLight
from . shader_projector import ShaderProjector



def test_illuminated_points():
    """
    To find the set of illuminated points (or shading points), we use the
    intersection points P of the first raytracer run and start a secondary
    raytracing from P towards the light source. If this secondary
    raytracing intersects with the mesh between P and the light source, the
    point is not illuminated by the light source.
    Unfortunately P is located on the mesh, sometimes some epsilon above or
    below the mesh; this means that the secondary raytracing may yield points
    very close to P, so the detection of illuminated points is almost random.
    There are two ways to solve this:
        * Filter the raytracer results for intersections very close to the origin
      of the ray ("scale" close to zero); in the Python implementation this is
      simple, but in the Embree implementation we have limited access to the
      API via Open3D
    * Move P slightly above the mesh using the triangle normal vectors; it is
      a bit of a hack but seems to work
        This test case checks if the shadow point detection works properly
    """
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



def test_shader_color():
    """ Put plane of different colors in front of a camera, take pictures
    using a parallel light shader (should give uniformly colored image)
    and check if color taken by camera have the expected color
    """
    # Cam
    cam = CameraModel(chip_size=(60, 45),
                      focal_length=(120, 90),
                    )
    cam_pose = Trafo3d(t=(0, 0, 300), rpy=np.deg2rad((180, 0, 0)))
    cam.set_pose(cam_pose)
    # Object: Plane that covers the full view of the camera
    plane = mesh_generate_plane((400, 400), color=(0.5, 0, 0))
    plane.translate(-plane.get_center())
    # Visualize scene
    if False:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        o3d.visualization.draw_geometries([world_cs, plane, cam_cs, cam_frustum])
    # Snap picture
    parallel_light = ShaderParallelLight(light_direction=(0, 0, -1))
    test_colors = ( (0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (0.2, 0.5, 0.7) )
    for test_color in test_colors:
        plane.paint_uniform_color(test_color)
        _, color_image, _ = cam.snap(plane, [ parallel_light ])
        assert np.all(np.isclose(color_image[:, :, 0], test_color[0])) # R
        assert np.all(np.isclose(color_image[:, :, 1], test_color[1])) # G
        assert np.all(np.isclose(color_image[:, :, 2], test_color[2])) # B



@pytest.mark.skip(reason="work in progess")
def test_projector_shader_color_blending():
    """ Put plane of different colors in front of a camera, take pictures
    using a projector shader with solid color and check if color taken by
    camera have the expected color
    """
    # Cam
    cam = CameraModel(chip_size=(60, 45),
                      focal_length=(120, 90),
                    )
    cam_pose = Trafo3d(t=(0, 0, 300), rpy=np.deg2rad((180, 0, 0)))
    cam.set_pose(cam_pose)
    # Object: Plane that covers the full view of the camera
    plane = mesh_generate_plane((400, 400), color=(0.5, 0, 0))
    plane.translate(-plane.get_center())
    # Projector
    image = np.zeros((45, 60, 3), dtype=np.uint8)
    projector = ShaderProjector(image, focal_length=(120, 90), pose=cam_pose)
    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        projector_cs = projector.get_cs(size=30.0)
        projector_frustum = projector.get_frustum(size=70.0)
        o3d.visualization.draw_geometries([world_cs, plane, cam_cs, cam_frustum,
            projector_cs, projector_frustum])
    # Define tests
    test_colors = [
        # object color   light color      expected resulting color
        [ [1, 1, 1],     [ 1, 1, 1 ],    [ 0.5, 0, 0 ] ],
    ]
    # Run tests
    for ocolor, lcolor, ecolor in test_colors:
        plane.paint_uniform_color(ocolor)
        image = np.tile(np.array(lcolor, dtype=np.uint8), (45, 60, 1))
        image = np.ones((45, 60, 3), dtype=np.uint8)
        projector.set_image(image)
        _, color_image, _ = cam.snap(plane, [ projector ])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(color_image)
        plt.show()

