import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load_multiple
from common.pixel_matcher import ImageMatcher
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector



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
    # For detection we can use any board, all in boards should be the same except
    # for the pose of the board
    board = boards[0]

    # Load images
    images = []
    for board_no in range(len(boards)):
        cam_images = []
        for cam_no in range(len(cams)):
            filenames = os.path.join(data_dir, \
                f'board{board_no:04}_cam{cam_no:04}_image????.png')
            cam_images.append(image_load_multiple(filenames))
        images.append(cam_images)

    print('Matching camera to projector pixels ...')
    matches = []
    for cam_no in range(len(cams)):
        board_matches = []
        for board_no in range(len(boards)):
            print(f'Matching pixels of cam{cam_no} image of board{board_no} to projector pixels ...')
            m = matcher.match(images[board_no][cam_no])
            board_matches.append(m)
        board_matches = np.asarray(board_matches)
        matches.append(board_matches)

    # Save results
    filename = os.path.join(data_dir, 'matches.npz')
    np.savez(filename, *matches)
