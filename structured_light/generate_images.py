import copy
import json
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_3float_to_rgb, image_save
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector



def visualize_scene(mesh, projector, cams):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ mesh ]
    objects.append(projector.get_cs(size=100))
    objects.append(projector.get_frustum(size=200))
    for cam in cams:
        objects.append(cam.get_cs(size=50))
        objects.append(cam.get_frustum(size=200))
    o3d.visualization.draw_geometries(objects)



if __name__ == '__main__':
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Generate mesh of object
    mesh = o3d.io.read_triangle_mesh('../data/fox_head.ply')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    mesh.scale(180, center=(0, 0, 0))
    #mesh.paint_uniform_color((1.0, 1.0, 1.0))
    mesh_pose = Trafo3d(t=(0, 0, 650), rpy=np.deg2rad((0, 160, 0)))
    mesh.transform(mesh_pose.get_homogeneous_matrix())

    # Generate projector
    projector_shape = (600, 800)
    projector_image = np.zeros((*projector_shape, 3))
    projector = ShaderProjector(image=projector_image,
        focal_length=1.2*np.asarray(projector_shape))
    projector_pose = Trafo3d(t=(0, 30, 0), rpy=np.deg2rad((10, 0, 0)))
    projector.set_pose(projector_pose)

    # Generate cameras
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam0.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam0_pose = Trafo3d(t=(-200, 10, 0), rpy=np.deg2rad((3, 16, 1)))
    cam0.set_pose(cam0_pose)
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam1.set_distortion((0.1, 0.05, 0.0, 0.05, -0.1, 0.12))
    cam1_pose = Trafo3d(t=(210, -5, 3), rpy=np.deg2rad((2, -14, -2)))
    cam1.set_pose(cam1_pose)
    cams = [ cam0, cam1 ]

    # Visualize scene
    visualize_scene(mesh, projector, cams)

    # Save configuration
    basename = os.path.join(data_dir, f'projector')
    projector.json_save(basename + '.json')
    for i, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'cam{i:02d}')
        cam.json_save(basename + '.json')
