import os
import sys
import time
import glob

import numpy as np
import matplotlib.pyplot as plt

import cv2

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load
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
    cpoints, ppoints):
    # Get indices in a circular region around the corner of the calibration
    # board given by the image point of the camera
    ci = circle_indices + cam_img_point.astype(int)
    # Get cam points and projector points of that region
    # TODO: This may go wrong if the circle radius is large and the
    # cam_img_point is located close to the boundaries of the
    # cpoints/ppoints arrays... so check boundaries?
    cp = cpoints[ci[:, 1], ci[:, 0], :]
    pp = ppoints[ci[:, 1], ci[:, 0], :]
    # Filter both by validity of projector points
    mask = np.all(np.isfinite(pp), axis=1)
    cp = cp[mask, :]
    pp = pp[mask, :]
    if cp.shape[0] < 4:
        raise Exception('Not enough points to calculate homography')
    # Calculate local homography, methods: 0, cv2.RANSAC, cv2.LMEDS, cv2.RHO
    H, mask = cv2.findHomography(cp, pp, method=cv2.LMEDS)
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
    # For detection we can use any board, all in boards should be the same except
    # for the pose of the board
    board = boards[0]
    filename = os.path.join(data_dir, f'matches.npz')
    npz = np.load(filename)
    image_points = npz['image_points']
    matches = []
    for cam_no in range(len(cams)):
        matches.append(npz[f'arr_{cam_no}'])



    circle_indices = generate_circle_indices(radius=10)
    if True:
        """ Plot one image with camera image point and region used
        to calculate local homography
        """
        cam_no = 0
        board_no = 0
        point_no = 7
        filename = os.path.join(data_dir, \
                f'board{board_no:04}_cam{cam_no:04}_image0001.png')
        white_image = image_load(filename)
        cam_img_point = image_points[cam_no, board_no, point_no, :]
        ci = circle_indices + cam_img_point.astype(int)
        _, ax = plt.subplots()
        ax.imshow(white_image)
        ax.plot(ci[:, 0], ci[:, 1], '.r', alpha=0.2)
        ax.plot(cam_img_point[0], cam_img_point[1], '+g')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'Image of cam{cam_no}, board{board_no}, point{point_no}')
        plt.show()



    all_projector_image_points = np.empty((len(cams), len(boards), board.max_num_points(), 2))
    all_projector_image_points[:] = np.NaN
    for cam_no in range(len(cams)):
        # Camera points
        shape = cams[cam_no].get_chip_size()[[1,0]]
        rows = np.arange(shape[0])
        cols = np.arange(shape[1])
        rows, cols = np.meshgrid(rows, cols, indexing='ij')
        cindices = np.vstack((rows.flatten(), cols.flatten())).T
        cpoints = image_indices_to_points(cindices)
        cpoints = cpoints.reshape((*shape, 2))
        for board_no in range(len(boards)):
            # Projector points
            pindices = matches[cam_no][board_no].reshape((-1, 2))
            ppoints = image_indices_to_points(pindices)
            ppoints = ppoints.reshape(matches[cam_no][board_no].shape)
            assert ppoints.shape == cpoints.shape
            for point_no in range(board.max_num_points()):
                cam_img_point = image_points[cam_no, board_no, point_no, :]
                if not np.all(np.isfinite(cam_img_point)):
                    continue
                proj_img_point = transform_cam_point_to_proj_point(cam_img_point,
                    circle_indices, cpoints, ppoints)
                all_projector_image_points[cam_no, board_no, point_no, :] = proj_img_point
    # Average over all cameras to get final projector image points
    projector_image_points = np.nanmedian(all_projector_image_points, axis=0)
    # Calculate errors, shape (num_cams, num_boards, num_corners)
    errors = all_projector_image_points - projector_image_points[np.newaxis, :, :, :]
    errors = np.sqrt(np.sum(np.square(errors), axis=3)) # Calculate distance on chip



    print('Errors per cam:')
    for cam_no in range(len(cams)):
        error_rms = np.sqrt(np.nanmean(np.square(errors[cam_no, :, :])))
        print(f'    cam{cam_no}: {error_rms:.3f} pixel RMS')
    print('Errors per board:')
    for board_no in range(len(boards)):
        error_rms = np.sqrt(np.nanmean(np.square(errors[:, board_no, :])))
        print(f'    board{board_no}: {error_rms:.3f} pixel RMS')



    if True:
        """ For a single board, plot all the camera image points for all cameras
        to the projector chip and visualize if the homography from both cameras work:
        projecting the same camera image point from different cameras to the same
        point on the projector chip
        """
        board_no = 8
        _, ax = plt.subplots()
        cs = projector.get_chip_size()
        ax.set_xlim(0, cs[0])
        ax.set_ylim(0, cs[1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        error_rms = np.sqrt(np.nanmean(np.square(errors[:, board_no, :])))
        ax.set_title(f'Projector chip with transformed points for board{board_no}, error {error_rms:.3f} pixel RMS')
        colors = [ 'r', 'g', 'b', 'c', 'm', 'y' ]
        for cam_no in range(len(cams)):
            pp = all_projector_image_points[cam_no, board_no, :, :]
            ax.plot(pp[:, 0], pp[:, 1], '+', color=colors[cam_no], label=f'cam{cam_no}')
        pp = projector_image_points[board_no, :, :]
        ax.plot(pp[:, 0], pp[:, 1], '+k', label='final')
        ax.grid()
        ax.legend()
        plt.show()

