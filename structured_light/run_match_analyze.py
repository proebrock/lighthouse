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
    cam_no = 0

    counters = np.zeros(projector_shape, dtype=int)
    indices = all_indices[cam_no]
    valid_mask = np.all(np.isfinite(indices), axis=2)
    indices = indices[valid_mask]
    indices = np.round(indices).astype(int)
    # TODO: Filter invalid indices already in matches in pixel_matcher?
    indices[:, 0] = np.clip(0, projector_shape[0]-1, indices[:, 0])
    indices[:, 1] = np.clip(0, projector_shape[1]-1, indices[:, 1])
    np.add.at(counters, tuple(indices.T), 1)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(counters)
    ax.set_title('Projector')
    ax = fig.add_subplot(122)
    ax.imshow(valid_mask)
    ax.set_title(f'Camera {cam_no}')
    plt.show()
