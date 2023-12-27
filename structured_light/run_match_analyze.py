import os
import glob

import numpy as np
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, '2d_calibrate_extrinsics')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    # Load matches
    pattern = os.path.join(data_dir, 'matches_cam????.npz')
    filenames = sorted(glob.glob(pattern))
    all_indices = []
    for filename in filenames:
        npz = np.load(filename)
        all_indices.append(npz['indices'])

    projector_shape = (600, 800) # TODO: load projector from json
    cam_shape = (900, 1200) # TODO: load projector from json
    cam_no = 0

    # Projector indices, shape (n, 2)
    pindices = all_indices[cam_no]
    valid_mask = np.all(np.isfinite(pindices), axis=2)
    # Sometimes matcher returns indices in subpixel accuracy that are
    # slightly out of bounds of projector chip indices
    valid_mask &= np.round(pindices[:, :, 0]) >= 0.0
    valid_mask &= np.round(pindices[:, :, 0]) <= (projector_shape[0] - 1)
    valid_mask &= np.round(pindices[:, :, 1]) >= 0.0
    valid_mask &= np.round(pindices[:, :, 1]) <= (projector_shape[1] - 1)
    pindices = pindices[valid_mask]
    pindices = np.round(pindices).astype(int)

    # Camera indices, shape (n, 2)
    cindices = np.zeros((*cam_shape, 2), dtype=int)
    i0 = np.arange(cam_shape[0])
    i1 = np.arange(cam_shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    cindices[:, :, 0] = i0
    cindices[:, :, 1] = i1
    cindices = cindices[valid_mask]

    counters = np.zeros(projector_shape, dtype=int)
    np.add.at(counters, tuple(pindices.T), 1)

    reverse_matches = {}
    for pi, ci in zip(pindices, cindices):
        _pi = tuple(pi.tolist())
        _ci = tuple(ci.tolist())
        if _pi not in reverse_matches:
            reverse_matches[_pi] = [ _ci ]
        else:
            reverse_matches[_pi].append(_ci)


    pfig, pax = plt.subplots()
    pax.set_title('Projector')
    pax.imshow(counters)

    cimage_base = np.zeros((*cam_shape, 3), dtype=np.uint8)
    cimage_base[valid_mask] = (0, 255, 0) # Green
    cimage_base[~valid_mask] = (0, 0, 0) # Black

    cfig, cax = plt.subplots()
    cax.set_title(f'Camera {cam_no}')
    img_handle = cax.imshow(cimage_base.copy())

    def pfig_mouse_move(event):
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        row = np.round(y).astype(int)
        col = np.round(x).astype(int)
        pidx = (row, col)
        if pidx not in reverse_matches:
            return
        cidx = reverse_matches[pidx]
        cidx = np.array(cidx)
        img = cimage_base.copy()
        img[cidx[:, 0], cidx[:, 1], :] = (255, 0, 0) # Red
        img_handle.set_data(img)
        cfig.canvas.draw_idle()



    def pfig_close_event(event):
        plt.close(cfig)

    pfig.canvas.mpl_connect('motion_notify_event', pfig_mouse_move)
    pfig.canvas.mpl_connect('close_event', pfig_close_event)
    plt.show(block=True)
