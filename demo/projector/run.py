import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

sys.path.append(os.path.abspath('../../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.shader_ambient_light import ShaderAmbientLight
from camsimlib.shader_point_light import ShaderPointLight
from camsimlib.shader_projector import ShaderProjector
from camsimlib.o3d_utils import mesh_generate_plane, show_images



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(100, 100),
                    )
    #cam.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam.scale_resolution(20)
    cam.place((-500, 0, 500))
    cam.look_at((0, 0, 0))
    cam.roll(np.deg2rad(-90))

    # Image file displayed by projector
    projector_image_filename = '../../data/lena.jpg'
    projector_image = cv2.imread(projector_image_filename, cv2.IMREAD_COLOR)
    if projector_image is None:
        raise Exception(f'Unable to read image file "{projector_image_filename}"')
    projector_image = cv2.cvtColor(projector_image, cv2.COLOR_BGR2RGB)
    projector_image = projector_image.astype(float) / 255
    projector_image = np.hstack((projector_image, projector_image[:,:,2::-1]))

    # Shaders
    ambient_light = ShaderAmbientLight(max_intensity=0.1)
    point_light = ShaderPointLight(light_position=(-500, 0, 270),
        max_intensity=0.1)
    projector = ShaderProjector(image=projector_image,
        focal_length=(2000, 1000))
    projector.place((-600, 0, 100))
    projector.look_at((0, 0, 0))
    projector.roll(np.deg2rad(-90))

    # Object
    # Simple plane
    plane = mesh_generate_plane((800, 800), color=(1, 1, 1))
    plane.compute_triangle_normals()
    plane.compute_vertex_normals()
    plane.translate(-plane.get_center())
    T = Trafo3d(t=(500, 0, 0), rpy=np.deg2rad((0, -70, 0)))
    plane.transform(T.get_homogeneous_matrix())

    # More complex object: Fox head
    fox = o3d.io.read_triangle_mesh('../../data/fox_head.ply')
    fox.compute_triangle_normals()
    fox.compute_vertex_normals()
    fox.translate(-fox.get_center())
    fox.scale(200, center=(0, 0, 0))
    fox.paint_uniform_color((1.0, 1.0, 1.0))

    mesh = plane + fox

    # Visualize scene
    if True:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        point_light_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=30)
        point_light_sphere.translate(point_light.get_light_position())
        point_light_sphere.paint_uniform_color((1, 1, 0))
        proj_cs = projector.get_cs(size=50.0)
        proj_frustum = projector.get_frustum(size=100.0)
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        o3d.visualization.draw_geometries([world_cs, point_light_sphere, mesh, \
            cam_cs, cam_frustum, proj_cs, proj_frustum])

    # Snap image
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(mesh, \
        shaders=[ambient_light, point_light, projector])
    toc = time.monotonic()
    print(f'Snapping image took {(toc - tic):.1f}s')

    # Show raw image
    fig, ax = plt.subplots()
    ax.imshow(projector_image)
    ax.set_axis_off()

    # Show resulting image
    fig, ax = plt.subplots()
    nanidx = np.where(np.isnan(color_image))
    img = color_image.copy()
    img[nanidx[0], nanidx[1], :] = (0, 1, 1)
    ax.imshow(img)
    ax.set_axis_off()
    plt.show()

    #show_images(depth_image, color_image)
    o3d.visualization.draw_geometries([pcl])
