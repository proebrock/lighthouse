import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2.aruco as aruco

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_show_multiple, \
    image_3float_to_rgb, image_save
from common.pixel_matcher import LineMatcherPhaseShift, ImageMatcher
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel
from camsimlib.shader_ambient_light import ShaderAmbientLight
from camsimlib.shader_projector import ShaderProjector



def mesh_black_to_gray(mesh, gray_value=0.3):
    colors = np.asarray(mesh.vertex_colors)
    mask_is_black = np.all(np.isclose(colors, 0.0), axis=1)
    colors[mask_is_black] = (gray_value, gray_value, gray_value)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)



def generate_board_poses(num_poses):
    rng = np.random.default_rng(0)
    translations = np.empty((num_poses, 3))
    translations[:,0] = rng.uniform(-50, 50, num_poses) # X
    translations[:,1] = rng.uniform(-50, 50, num_poses) # Y
    translations[:,2] = rng.uniform(-100, 400, num_poses) # Z
    rotations_rpy = np.empty((num_poses, 3))
    rotations_rpy[:,0] = rng.uniform(-10, 10, num_poses) # X
    rotations_rpy[:,1] = rng.uniform(-10, 10, num_poses) # Y
    rotations_rpy[:,2] = rng.uniform(-10, 10, num_poses) # Z
    rotations_rpy = np.deg2rad(rotations_rpy)
    return [ Trafo3d(t=translations[i,:],
                     rpy=rotations_rpy[i,:]) for i in range(num_poses)]



def visualize_scene(meshes, projector, cams):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ *meshes ]
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

    # Prepare scene: Chessboard and meshes
    board_squares = (7, 5)
    board_square_length_pix = 80
    board_square_length_mm = 30.0
    board_marker_length_mm = 15.0
    board_pose = Trafo3d(t=(-100, -100, 500), rpy=np.deg2rad((0, 0, 0)))
    boards = []
    meshes = []
    for trafo in generate_board_poses(12):
        board = CharucoBoard(board_squares, board_square_length_pix,
            board_square_length_mm, board_marker_length_mm,
            aruco.DICT_4X4_100, pose=board_pose*trafo)
        boards.append(board)
        screen = board.generate_screen()
        mesh = screen.get_mesh()
        mesh_black_to_gray(mesh)
        meshes.append(mesh)

    # Generate projector
    projector_shape = (600, 800)
    projector_image = np.zeros((*projector_shape, 3), dtype=np.uint8)
    projector = ShaderProjector(image=projector_image,
        focal_length=0.9*np.asarray(projector_shape))
    #projector.set_distortion((-0.05, 0.1, 0.1, -0.05, 0.25, 0.07))

    # Generate cameras
    cam0 = CameraModel(chip_size=(32, 20), focal_length=(32, 32))
    #cam0.set_distortion((-0.1, 0.1, 0.05, -0.05, 0.2, 0.08))
    cam0_pose = Trafo3d(t=(-200, 10, 0), rpy=np.deg2rad((3, 16, 1)))
    cam0.set_pose(cam0_pose)
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40))
    #cam1.set_distortion((0.1, 0.05, 0.0, 0.05, -0.1, 0.12))
    cam1_pose = Trafo3d(t=(210, -5, 3), rpy=np.deg2rad((2, -14, -2)))
    cam1.set_pose(cam1_pose)
    cams = [ cam0, cam1 ]
    for cam in cams:
        cam.scale_resolution(40)

    # Visualize scene
    #visualize_scene(meshes, projector, cams)

    # Generate projector images
    num_time_steps = 7
    num_phases = 2
    row_matcher = LineMatcherPhaseShift(projector_shape[0],
        num_time_steps, num_phases)
    col_matcher = LineMatcherPhaseShift(projector_shape[1],
        num_time_steps, num_phases)
    matcher = ImageMatcher(projector_shape, row_matcher, col_matcher)
    images = matcher.generate()
    #image_show_multiple(images, single_window=True)
    #plt.show()

    # Snap and save images
    ambient_light = ShaderAmbientLight(max_intensity=0.1)
    for mesh_no in range(len(meshes)):
        for cam_no in range(len(cams)):
            for image_no in range(images.shape[0]):
                basename = os.path.join(data_dir,
                    f'board{mesh_no:04}_cam{cam_no:04}_image{image_no:04}')
                print(f'Snapping image {basename} ...')
                cam = cams[cam_no]
                tic = time.monotonic()
                projector.set_image(images[image_no])
                _, cam_image, _ = cam.snap(meshes[mesh_no], \
                    shaders=[ambient_light, projector])
                toc = time.monotonic()
                print(f'Snapping image took {(toc - tic):.1f}s')
                # Save generated snap
                cam_image = image_3float_to_rgb(cam_image)
                image_save(basename + '.png', cam_image)

    # Save configuration
    filename = os.path.join(data_dir, 'projector.json')
    projector.json_save(filename)
    for i, cam in enumerate(cams):
        basename = os.path.join(data_dir, f'cam{i:02d}')
        cam.json_save(basename + '.json')
    for i, board in enumerate(boards):
        basename = os.path.join(data_dir, f'board{i:02d}')
        board.json_save(basename + '.json')
    filename = os.path.join(data_dir, 'matcher.json')
    matcher.json_save(filename)
