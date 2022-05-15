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
from camsimlib.shader_projector import ShaderProjector
from camsimlib.shader_point_light import ShaderPointLight
from camsimlib.o3d_utils import show_images



if __name__ == '__main__':
    # Camera
    cam = CameraModel(chip_size=(120, 90),
                      focal_length=(100, 100),
                    )
    #cam.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam.scale_resolution(1)
    cam.place((-500, 0, 500))
    cam.look_at((0, 0, 0))
    cam.roll(np.deg2rad(-90))

    # Image file displayed by projector
    projector_image_filename = '../../data/tux.png'
    projector_image = cv2.imread(projector_image_filename, cv2.IMREAD_COLOR)
    if projector_image is None:
        raise Exception(f'Unable to read image file "{projector_image_filename}"')
    projector_image = cv2.cvtColor(projector_image, cv2.COLOR_BGR2RGB)
    projector_image = projector_image.astype(float) / 255
    if False:
        fig, ax = plt.subplots()
        ax.imshow(projector_image)
        plt.show()

    # Shader implementing projector
    projector = ShaderProjector(image=projector_image, focal_length=(100, 100))
    projector.scale_resolution(1)
    projector.place((-600, 0, -200))
    projector.look_at((0, 0, 0))
    projector.roll(np.deg2rad(-90))
    #projector = ShaderPointLight(light_position=(-600, 0, -200))

    # Object
    mesh = o3d.io.read_triangle_mesh('../../data/fox_head.ply')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    mesh.scale(200, center=(0, 0, 0))
    mesh.paint_uniform_color((1.0, 1.0, 1.0))

    # Visualize scene
    if False:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200.0)
        proj_cs = projector.get_cs(size=50.0)
        proj_frustum = projector.get_frustum(size=100.0)
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=100.0)
        o3d.visualization.draw_geometries([world_cs, mesh, \
            cam_cs, cam_frustum, proj_cs, proj_frustum])

    # Snap image
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(mesh, shaders=[projector])
    toc = time.monotonic()
    print(f'Snapping image took {(toc - tic):.1f}s')

    # Visualize images and point cloud
    show_images(depth_image, color_image)
    #o3d.visualization.draw_geometries([cs, pcl])
