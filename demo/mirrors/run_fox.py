import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from common.mesh_utils import mesh_generate_surface
from camsimlib.camera_model import CameraModel
from camsimlib.shader_ambient_light import ShaderAmbientLight
from camsimlib.shader_point_light import ShaderPointLight
from camsimlib.multi_mesh import MultiMesh



def generate_frame(inner_width, outer_width):
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.array((
        ( outer_width/2.0,  outer_width/2.0, 0.0),
        (-outer_width/2.0,  outer_width/2.0, 0.0),
        (-outer_width/2.0, -outer_width/2.0, 0.0),
        ( outer_width/2.0, -outer_width/2.0, 0.0),
        ( inner_width/2.0,  inner_width/2.0, 0.0),
        (-inner_width/2.0,  inner_width/2.0, 0.0),
        (-inner_width/2.0, -inner_width/2.0, 0.0),
        ( inner_width/2.0, -inner_width/2.0, 0.0),
        ( inner_width/2.0,  outer_width/2.0, 0.0),
        (-outer_width/2.0,  inner_width/2.0, 0.0),
        (-inner_width/2.0, -outer_width/2.0, 0.0),
        ( outer_width/2.0, -inner_width/2.0, 0.0),
    ))
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    triangles = np.array((
        (0, 8, 11),
        (1, 9, 8),
        (2, 10, 9),
        (3, 11, 10),
        (4, 8, 9),
        (5, 9, 10),
        (6, 10, 11),
        (7, 11, 8),
    ), dtype=int)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(100, 100),
                    )
    cam.scale_resolution(20)
    cam.place((400, 0, -400))
    cam.look_at((0, 0, 0))
    cam.roll(np.deg2rad(90))

    # Mirror
    fun = lambda x: np.cos(x[:, 0]) * np.cos(x[:, 1])
    mirror = mesh_generate_surface(fun, xrange=(-np.pi/2, np.pi/2), yrange=(-np.pi/2, np.pi/2),
        num=(50, 50), scale=(360.0, 360.0, -25.0))
    mirror.translate((-180, -180, 0.0))
    T = Trafo3d(rpy=np.deg2rad((0, 180.0-45/2, 0)))
    mirror.transform(T.get_homogeneous_matrix())
    mirror.paint_uniform_color((1.0, 0.0, 0.0))

    # Mirror frame
    frame = generate_frame(360, 390)
    T = Trafo3d(rpy=np.deg2rad((0, 180.0-45/2, 0)))
    frame.transform(T.get_homogeneous_matrix())
    frame.paint_uniform_color((0.0, 0.0, 1.0))

    # Object
    fox = o3d.io.read_triangle_mesh('../../data/fox_head.ply')
    fox.translate(-fox.get_center())
    fox.scale(100.0, center=(0, 0, 0))
    fox.translate((0.0, 0.0, -250.0))
    fox.compute_triangle_normals()
    fox.compute_vertex_normals()

    # Shaders
    ambient_light = ShaderAmbientLight(max_intensity=0.1)
    point_light = ShaderPointLight(light_position=(300, 0, 0),
        max_intensity=0.8)

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100.0)
        cam_cs = cam.get_cs(50.0)
        cam_frustum = cam.get_frustum(200.0)
        point_light_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=30)
        point_light_sphere.translate(point_light.get_light_position())
        point_light_sphere.paint_uniform_color((1, 1, 0))
        o3d.visualization.draw_geometries([world_cs, cam_cs, cam_frustum, \
            point_light_sphere, fox, mirror, frame])

    # Snap image
    meshes = MultiMesh([fox, mirror, frame], [False, True, False])
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(meshes, \
        shaders=[ambient_light, point_light])
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

    # Visualize images and point cloud
    #show_images(depth_image, color_image)
    #o3d.visualization.draw_geometries([cam_cs, pcl])
