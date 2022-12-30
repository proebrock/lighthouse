import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.shader_ambient_light import ShaderAmbientLight
from camsimlib.shader_point_light import ShaderPointLight
from camsimlib.multi_mesh import MultiMesh
from camsimlib.o3d_utils import show_images



def generate_mirror(n, scale):
    mesh = o3d.geometry.TriangleMesh()
    # Generate vertices
    x = np.linspace(-np.pi/2, np.pi/2, n)
    y = np.linspace(-np.pi/2, np.pi/2, n)
    x, y = np.meshgrid(x, y, indexing='ij')
    z = scale * np.cos(x) * np.cos(y)
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    mesh.vertices = o3d.utility.Vector3dVector(points)
    # Generate triangles
    triangles = np.zeros((2 * (n - 1) * (n - 1), 3), dtype=int)
    index = 0
    for row in range(1, n):
        for col in range(1, n):
            triangles[index, 0] = (row - 0) + n * (col - 0)
            triangles[index, 1] = (row - 1) + n * (col - 0)
            triangles[index, 2] = (row - 0) + n * (col - 1)
            index += 1
            triangles[index, 0] = (row - 1) + n * (col - 1)
            triangles[index, 1] = (row - 0) + n * (col - 1)
            triangles[index, 2] = (row - 1) + n * (col - 0)
            index += 1
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # Calculate normals
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh



def visualize_mesh_with_normals(mesh):
    # Generate PCL
    pcl = o3d.geometry.PointCloud()
    points = np.asarray(mesh.vertices)
    pcl.points = o3d.utility.Vector3dVector(points)
    normals = np.asarray(mesh.vertex_normals)
    pcl.normals = o3d.utility.Vector3dVector(normals)
    # Visualize
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcl)
    vis.add_geometry(mesh)
    vis.get_render_option().point_show_normal = True
#    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Normal
    vis.run()
    vis.destroy_window()



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(100, 100),
                    )
    cam.scale_resolution(20)
    cam.place((500, 0, -500))
    cam.look_at((0, 0, 0))
    cam.roll(np.deg2rad(90))
    cam_cs = cam.get_cs(50.0)
    cam_frustum = cam.get_frustum(200.0)

    # Mirror
    mirror = generate_mirror(50, -0.2)
    #visualize_mesh_with_normals(mirror)
    T = Trafo3d(rpy=np.deg2rad((0, -30, 0)))
    mirror.transform(T.get_homogeneous_matrix())
    mirror.scale(100.0, center=(0, 0, 0))
    mirror.paint_uniform_color((1.0, 0.0, 0.0))

    # Object
    fox = o3d.io.read_triangle_mesh('../../data/fox_head.ply')
    fox.translate(-fox.get_center())
    fox.scale(100.0, center=(0, 0, 0))
    fox.translate((0.0, 0.0, -300.0))
    fox.compute_triangle_normals()
    fox.compute_vertex_normals()

    # Shaders
    ambient_light = ShaderAmbientLight(max_intensity=0.1)
    point_light = ShaderPointLight(light_position=(300, 0, 0),
        max_intensity=0.8)

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        point_light_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=30)
        point_light_sphere.translate(point_light.get_light_position())
        point_light_sphere.paint_uniform_color((1, 1, 0))
        o3d.visualization.draw_geometries([world_cs, cam_cs, cam_frustum, \
            point_light_sphere, fox, mirror])

    # Snap image
    meshes = MultiMesh([fox, mirror], [False, True])
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

    o3d.visualization.draw_geometries([cam_cs, pcl])
