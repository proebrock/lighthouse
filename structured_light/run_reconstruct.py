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



def cam_get_match_indices(indices, projector_shape, cam_shape):
    # Projector indices, shape (n, 2)
    # These are the target pixels of the matching
    pindices = indices.reshape((-1, 2))
    valid_mask = np.all(np.isfinite(pindices), axis=1)
    valid_mask &= pindices[:, 0] >= 0.0
    valid_mask &= pindices[:, 0] <= (projector_shape[0] - 1)
    valid_mask &= pindices[:, 1] >= 0.0
    valid_mask &= pindices[:, 1] <= (projector_shape[1] - 1)
    pindices = pindices[valid_mask]
    # Camera indices, shape (n, 2)
    # These are the source pixels of the matching
    cindices = np.zeros((*cam_shape, 2))
    i0 = np.arange(cam_shape[0])
    i1 = np.arange(cam_shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    cindices[:, :, 0] = i0
    cindices[:, :, 1] = i1
    cindices = cindices.reshape((-1, 2))
    cindices = cindices[valid_mask]
    # The i-th index of cindices maps the i-th index of pindices
    assert pindices.shape == cindices.shape
    return pindices, cindices



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
    all_indices = []
    for filename in filenames:
        npz = np.load(filename)
        all_indices.append(npz['indices'])

    match_indices = []
    projector_shape = projector.get_chip_size()[[1, 0]]
    for cam_no in range(len(cams)):
        cam_shape = cams[cam_no].get_chip_size()[[1, 0]]
        pindices, cindices = cam_get_match_indices(all_indices[cam_no],
            projector_shape, cam_shape)
        match_indices.append([pindices, cindices])
