import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_show_multiple, \
    image_3float_to_rgb, image_save
from common.mesh_utils import mesh_generate_plane, mesh_save
from common.pixel_matcher import LineMatcherPhaseShift, ImageMatcher
from camsimlib.camera_model import CameraModel
from camsimlib.shader_ambient_light import ShaderAmbientLight
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
    if False:
        # Simple test mesh: plane
        mesh = mesh_generate_plane((1000, 1000), color=(1, 1, 1))
        mesh.translate(-mesh.get_center())
        mesh_pose = Trafo3d(t=(0, 0, 650), rpy=np.deg2rad((10, 180, -5)))
    else:
        mesh = o3d.io.read_triangle_mesh('../data/fox_head.ply')
        mesh.translate(-mesh.get_center())
        mesh.scale(180, center=(0, 0, 0))
        mesh_pose = Trafo3d(t=(0, 0, 650), rpy=np.deg2rad((0, 160, 180)))
    mesh.transform(mesh_pose.get_homogeneous_matrix())
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # Generate projector
    projector_shape = (600, 800)
    projector_image = np.zeros((*projector_shape, 3), dtype=np.uint8)
    projector = ShaderProjector(image=projector_image,
        focal_length=0.9*np.asarray(projector_shape))
    projector_pose = Trafo3d(t=(0, 30, 0), rpy=np.deg2rad((10, 0, 0)))
    projector.set_pose(projector_pose)

    # Generate cameras
    cam0 = CameraModel(chip_size=(32, 20), focal_length=(32, 32))
    cam0.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam0_pose = Trafo3d(t=(-200, 10, 0), rpy=np.deg2rad((3, 16, 1)))
    cam0.set_pose(cam0_pose)
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    cam1.set_distortion((0.1, 0.05, 0.0, 0.05, -0.1, 0.12))
    cam1_pose = Trafo3d(t=(210, -5, 3), rpy=np.deg2rad((2, -14, -2)))
    cam1.set_pose(cam1_pose)
    cams = [ cam0, cam1 ]
    for cam in cams:
        cam.scale_resolution(40)

    # Visualize scene
    #visualize_scene(mesh, projector, cams)

    # Generate projector images
    num_time_steps = 11
    num_phases = 2
    row_matcher = LineMatcherPhaseShift(projector_shape[0],
        num_time_steps, num_phases)
    col_matcher = LineMatcherPhaseShift(projector_shape[1],
        num_time_steps, num_phases)
    matcher = ImageMatcher(projector_shape, row_matcher, col_matcher)
    images = matcher.generate()
    #image_show_multiple(images, single_window=True)
    #plt.show()

    # Snap camera images
    ambient_light = ShaderAmbientLight(max_intensity=0.1)
    for image_no in range(images.shape[0]):
        for cam_no in range(len(cams)):
            basename = os.path.join(data_dir,
                f'image{image_no:04}_cam{cam_no:04}')
            print(f'Snapping image {basename} ...')
            projector.set_image(images[image_no])
            cam = cams[cam_no]
            tic = time.monotonic()
            _, cam_image, _ = cam.snap(mesh, \
                shaders=[ambient_light, projector])
            toc = time.monotonic()
            print(f'Snapping image took {(toc - tic):.1f}s')
            # Save generated snap
            cam_image = image_3float_to_rgb(cam_image)
            image_save(basename + '.png', cam_image)

    # Save configuration
    filename = os.path.join(data_dir, 'mesh.ply')
    mesh_save(filename, mesh)
    filename = os.path.join(data_dir, 'projector.json')
    projector.json_save(filename)
    for i, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'cam{i:02d}')
        cam.json_save(basename + '.json')
    filename = os.path.join(data_dir, 'matcher.json')
    matcher.json_save(filename)
