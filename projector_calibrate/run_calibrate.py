import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import cv2

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load_multiple
from camsimlib.image_mapping import image_indices_to_points
from common.pixel_matcher import ImageMatcher
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector



def generate_circle_indices(radius):
    # Generate indices for rectangular region
    rows = np.arange(-radius, radius + 1)
    cols = np.arange(-radius, radius + 1)
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    indices = np.vstack((rows.flatten(), cols.flatten())).T
    # Reduce indices to circular region
    mask = np.sum(np.square(indices), axis=1) < (radius * radius)
    indices = indices[mask, :]
    return indices



def transform_cam_point_to_proj_point(cam_img_point, circle_indices,
    cpoints, ppoints, img=None):
    # Get indices in a circular region around the corner of the calibration
    # board given by the image point of the camera
    ci = circle_indices + cam_img_point.astype(int)
    if img is not None:
        _, ax = plt.subplots()
        ax.imshow(img)
        ax.plot(ci[:, 0], ci[:, 1], '.r', alpha=0.2)
        ax.plot(cam_img_point[0], cam_img_point[1], '+g')
        plt.show()
    # Get cam points and projector points of that region
    cp = cpoints[ci[:, 1], ci[:, 0], :]
    pp = ppoints[ci[:, 1], ci[:, 0], :]
    # Filter both by validity of projector points
    mask = np.all(np.isfinite(pp), axis=1)
    cp = cp[mask, :]
    pp = pp[mask, :]
    if cp.shape[0] < 4:
        raise Exception('Not enough points to calculate homography')
    # Calculate local homography, methods: 0, RANSAC, LMEDS, RHO
    H, mask = cv2.findHomography(cp, pp, method=0)
    # Translate camera image point to projector using homography
    x = H @ np.array((cam_img_point[0], cam_img_point[1], 1.0))
    if np.isclose(x[2], 0.0):
        raise Exception('Invalid homography')
    return np.array((x[0] / x[2], x[1] / x[2]))



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'projector_calibrate')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load configuration
    filename = os.path.join(data_dir, 'projector.json')
    projector = ShaderProjector()
    projector.json_load(filename)
    cam_filenames = sorted(glob.glob(os.path.join(data_dir, 'cam??.json')))
    cams = []
    for i, filename in enumerate(cam_filenames):
        cam = CameraModel()
        cam.json_load(filename)
        cams.append(cam)
    filename = os.path.join(data_dir, 'matcher.json')
    matcher = ImageMatcher()
    matcher.json_load(filename)
    board_filenames = sorted(glob.glob(os.path.join(data_dir, 'board??.json')))
    boards = []
    for i, filename in enumerate(board_filenames):
        board = CharucoBoard()
        board.json_load(filename)
        boards.append(board)
    filename = os.path.join(data_dir, f'matches.npz')
    npz = np.load(filename)
    object_points = npz['object_points']
    image_points = npz['image_points']
    matches = []
    for cam_no in range(len(cams)):
        matches.append(npz[f'arr_{cam_no}'])

    # Load images
    images = []
    for board_no in range(len(boards)):
        cam_images = []
        for cam_no in range(len(cams)):
            filenames = os.path.join(data_dir, \
                f'board{board_no:04}_cam{cam_no:04}_image????.png')
            cam_images.append(image_load_multiple(filenames))
        images.append(cam_images)

    circle_indices = generate_circle_indices(radius=10)

    cam_no = 0

    # Camera points
    shape = cams[cam_no].get_chip_size()[[1,0]]
    rows = np.arange(shape[0])
    cols = np.arange(shape[1])
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    cindices = np.vstack((rows.flatten(), cols.flatten())).T
    cpoints = image_indices_to_points(cindices)
    cpoints = cpoints.reshape((*shape, 2))

    board_no = 0

    # Projector points
    pindices = matches[cam_no][board_no].reshape((-1, 2))
    ppoints = image_indices_to_points(pindices)
    ppoints = ppoints.reshape(matches[cam_no][board_no].shape)
    assert ppoints.shape == cpoints.shape

    point_no = 10
    white_image = images[board_no][cam_no][1]
    img_point = image_points[cam_no, board_no, point_no, :]
    transform_cam_point_to_proj_point(img_point, circle_indices,
        cpoints, ppoints, white_image)

