import os
import sys
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from common.pixel_matcher import ImageMatcher
from common.mesh_utils import mesh_load
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector
from camsimlib.image_mapping import image_indices_to_points, \
    image_indices_on_chip_mask



def cam_get_match_indices(matches, projector, cam):
    # Projector indices, shape (n, 2)
    # these are the target pixels of the matching;
    # pixel matcher gives indices, not points
    pindices = matches.reshape((-1, 2))
    valid_mask = image_indices_on_chip_mask(pindices, projector.get_chip_size())
    pindices = pindices[valid_mask]
    # Camera points, shape (n, 2)
    # These are the source pixels of the matching
    rows = np.arange(cam.get_chip_size()[1])
    cols = np.arange(cam.get_chip_size()[0])
    rows, cols = np.meshgrid(rows, cols, indexing='ij')
    cindices = np.vstack((rows.flatten(), cols.flatten())).T
    cindices = cindices[valid_mask]
    return pindices, cindices, valid_mask.reshape((cam.get_chip_size()[[1, 0]]))



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'structured_light')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load configuration
    filename = os.path.join(data_dir, 'mesh.ply')
    mesh = mesh_load(filename)
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

    # Load matches
    pattern = os.path.join(data_dir, 'matches_cam????.npz')
    filenames = sorted(glob.glob(pattern))
    all_matches = []
    for filename in filenames:
        npz = np.load(filename)
        all_matches.append(npz['matches'])

    cam_no = 0
    cam_shape = cams[cam_no].get_chip_size()[[1, 0]]
    projector_shape = projector.get_chip_size()[[1, 0]]

    pindices, cindices, valid_mask = \
        cam_get_match_indices(all_matches[cam_no],
        projector, cams[cam_no])
    # We round the projector pixels; with this we loose
    # subpixel accuracy but this does not matter in a rough
    # analysis of the situation
    pindices = np.round(pindices).astype(int)
    cindices = np.round(cindices).astype(int)

    # For each projector pixel count the number of
    # camera pixels matching to this projector pixel
    counters = np.zeros(projector_shape, dtype=int)
    np.add.at(counters, tuple(pindices.T), 1)
    print(f'These matches target {np.sum(counters > 0)} projector pixels')
    print(f'Of those {np.sum(counters > 1)} are matched by more than one cam pixel')

    # Reverse matching: key of dict is projector pixel,
    # value is list of camera pixels matching to this
    # projector pixel
    reverse_matches = {}
    for pi, ci in zip(pindices, cindices):
        _pi = tuple(pi.tolist())
        _ci = tuple(ci.tolist())
        if _pi not in reverse_matches:
            reverse_matches[_pi] = [ _ci ]
        else:
            reverse_matches[_pi].append(_ci)

    if True:
        # For each projector pixel plot count of cam pixels
        # matching to this single projector pixel
        pfig, pax = plt.subplots()
        pax.set_title('Projector')
        pimage = counters.astype(float)
        pimage[counters == 0] = np.NaN
        cmap = mpl.colormaps.get_cmap('viridis')
        cmap.set_bad(color='c') # Cyan
        pax.imshow(pimage, cmap=cmap)
        # For each camera pixel plot
        cfig, cax = plt.subplots()
        cax.set_title(f'Camera {cam_no}')
        cimage_base = np.zeros((*cam_shape, 3), dtype=np.uint8)
        cimage_base[valid_mask] = (255, 255, 0) # Green
        cimage_base[~valid_mask] = (0, 255, 255) # Cyan
        img_handle = cax.imshow(cimage_base.copy())

        def pfig_mouse_move(event):
            x, y = event.xdata, event.ydata
            if x is None or y is None:
                return
            row = np.round(y).astype(int)
            col = np.round(x).astype(int)
            pidx = (row, col)
            img = cimage_base.copy()
            if pidx in reverse_matches:
                cidx = reverse_matches[pidx]
                cidx = np.array(cidx)
                img[cidx[:, 0], cidx[:, 1], :] = (255, 0, 0) # Red
            img_handle.set_data(img)
            cfig.canvas.draw_idle()

        def pfig_close_event(event):
            plt.close(cfig)

        def cfig_close_event(event):
            plt.close(pfig)

        pfig.canvas.mpl_connect('motion_notify_event', pfig_mouse_move)
        pfig.canvas.mpl_connect('close_event', pfig_close_event)
        cfig.canvas.mpl_connect('close_event', cfig_close_event)
        plt.show(block=True)


    if True:
        # Define ROI to limit amount of points
        p_rows_minmax = (250, 370)
        p_cols_minmax = (320, 440)

        cam_points = []
        for cam_no in range(len(cams)):
            indices = all_matches[cam_no].reshape((-1, 2))
            points = image_indices_to_points(indices)
            valid_mask = projector.points_on_chip_mask(points)
            cam_points.append(points[valid_mask])

        fig, ax = plt.subplots()
        colors = [ 'r', 'g', 'b', 'c', 'm', 'y' ]
        for cam_no in range(len(cams)):
            points = cam_points[cam_no]
            ax.plot(points[:, 0], points[:, 1], '+', color=colors[cam_no],
                label=f'cam{cam_no}')
        ax.set_title('Projector chip ROI')
        if False:
            # View pixel borders
            ax.xaxis.set_ticks(np.arange(p_rows_minmax[0], p_rows_minmax[1] + 1))
            ax.yaxis.set_ticks(np.arange(p_cols_minmax[0], p_cols_minmax[1] + 1))
            ax.set_aspect('equal')
            ax.grid()
        ax.legend()
        plt.show()

