import os
import sys
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.pixel_matcher import ImageMatcher
from common.mesh_utils import mesh_load
from camsimlib.camera_model import CameraModel
from camsimlib.shader_projector import ShaderProjector
from common.bundle_adjust import bundle_adjust_points



def cam_get_match_points(matches, projector_shape, cam_shape):
    # Projector points, shape (n, 2)
    # These are the target pixels of the matching
    ppoints = matches.reshape((-1, 2))
    valid_mask = np.all(np.isfinite(ppoints), axis=1)
    valid_mask &= ppoints[:, 0] >= 0.0
    valid_mask &= ppoints[:, 0] <= (projector_shape[0] - 1)
    valid_mask &= ppoints[:, 1] >= 0.0
    valid_mask &= ppoints[:, 1] <= (projector_shape[1] - 1)
    ppoints = ppoints[valid_mask]
    # Camera points, shape (n, 2)
    # These are the source pixels of the matching
    cpoints = np.zeros((*cam_shape, 2))
    i0 = np.arange(cam_shape[0])
    i1 = np.arange(cam_shape[1])
    i0, i1 = np.meshgrid(i0, i1, indexing='ij')
    cpoints[:, :, 0] = i0
    cpoints[:, :, 1] = i1
    cpoints = cpoints.reshape((-1, 2))
    cpoints = cpoints[valid_mask]
    # The i-th index of cpoints maps the i-th index of ppoints
    assert ppoints.shape == cpoints.shape
    return ppoints, cpoints, valid_mask.reshape((cam_shape))



def cluster_points(match_points):
    num_cams = len(match_points)
    reverse_matches = {}
    for cam_no, (ppoints, cpoints, valid_mask) in enumerate(match_points):
        pindices = np.round(ppoints).astype(int)
        for pi, pp, cp in zip(pindices, ppoints, cpoints):
            _pi = tuple(pi.tolist())
            _pp = tuple(pp.tolist())
            _cp = tuple(cp.tolist())
            if _pi not in reverse_matches:
                new_entry = [ [ _pp, ], ]
                for _ in range(num_cams):
                    new_entry.append([])
                reverse_matches[_pi] = new_entry
            reverse_matches[_pi][cam_no+1].append(_cp)
    return list(reverse_matches.values())



def create_bundle_adjust_points(reverse_matches):
    num_points = len(reverse_matches)
    num_cams = len(reverse_matches[0])
    points = np.zeros((num_points, num_cams, 2))
    points[:] = np.NaN
    for point_no in range(num_points):
        for cam_no in range(num_cams):
            p = np.array(reverse_matches[point_no][cam_no])
            if p.size == 0:
                continue
            center = np.mean(p, axis=0)
            assert center.size == 2
            points[point_no, cam_no, :] = center
    return points



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
    all_matches = []
    for filename in filenames:
        npz = np.load(filename)
        all_matches.append(npz['matches'])

    match_points = []
    projector_shape = projector.get_chip_size()[[1, 0]]
    for cam_no in range(len(cams)):
        cam_shape = cams[cam_no].get_chip_size()[[1, 0]]
        ppoints, cpoints, valid_mask = cam_get_match_points(all_matches[cam_no],
            projector_shape, cam_shape)
        match_points.append([ppoints, cpoints, valid_mask])

    print('Reverse matching ...')
    reverse_matches = cluster_points(match_points)
    print('Creating bundle adjust points ...')
    p = create_bundle_adjust_points(reverse_matches)

    enough_mask = np.sum(np.isfinite(p[:, :, 0]), axis=1) >= 2
    p = p[enough_mask, :, :]

    bundle_adjust_cams = [ projector, ]
    bundle_adjust_cams.extend(cams)
    print('Running bundle adjustment ...')
    P = bundle_adjust_points(bundle_adjust_cams, p)
    print('Done.')

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(P)
    pcl.paint_uniform_color((1, 0, 0))
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    o3d.visualization.draw_geometries([cs, pcl])


