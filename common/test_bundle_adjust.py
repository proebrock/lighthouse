import numpy as np
np.random.seed(42)
import pytest

import open3d as o3d

from trafolib.trafo3d import Trafo3d
from common.bundle_adjust import bundle_adjust
from camsimlib.camera_model import CameraModel



def visualize_scene(cams, P):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    objects = [ cs ]
    for cam in cams:
        objects.append(cam.get_cs(size=50))
        objects.append(cam.get_frustum(size=200))
    for i in range(P.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=5)
        sphere.paint_uniform_color((1, 0, 0))
        sphere.translate(P[i, :])
        objects.append(sphere)
    o3d.visualization.draw_geometries(objects)



def scene_to_chip(cams, P):
    points = np.zeros((P.shape[0], len(cams), 2))
    for i, cam in enumerate(cams):
        p = cam.scene_to_chip(P)
        p = p[:, 0:2] # Omit distances
        points[:, i, :] = p
    return points



def test_variying_visibility():
    # Setup scene
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(200, 0 ,0)))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(-200, 0 ,0)))
    cam2 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(0, 200 ,0)))
    cam3 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(0, -200, 0)))
    cams = [ cam0, cam1, cam2, cam3 ]
    P = np.array((
        (-100, 200, 800),
        (100, 0, 800),
        (-100, 200, 600),
        (50, -50, 900),
        ))
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)

    # Disable some observations
    visible_mask = np.array((
        (False, False, False, False), # Point 0 visibile by 0 cameras
        (True,  False, False, False), # Point 1 visibile by 1 cameras
        (True,  False, True,  False), # Point 2 visibile by 2 cameras
        (True,  True,  True,  False), # Point 3 visibile by 3 cameras
    ), dtype=bool)                    # camera 3 does not see anything
    p[~visible_mask, :] = np.NaN

    # Check bundle adjust with varying visibilities; if not enough observations
    # available, we expect exceptions
    with pytest.raises(ValueError):
        bundle_adjust(cams, p[0:4, :, :])
    with pytest.raises(ValueError):
        bundle_adjust(cams, p[1:4, :, :])
    bundle_adjust(cams, p[3:4, :, :])
    bundle_adjust(cams, p[2:4, :, :], full=True)



def test_residuals():
    # Setup scene
    cam0 = CameraModel(chip_size=(800, 600), focal_length=(800, 800),
        pose=Trafo3d(t=(250, 0 ,0)))
    cam1 = CameraModel(chip_size=(800, 600), focal_length=(800, 800),
        pose=Trafo3d(t=(-250, 0 ,0)))
    cams = [ cam0, cam1 ]
    P = np.array(((0, 0, 1000),))
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)
    # Introduce error of d pixels in opposite directions
    d = 10.0
    p[0, 0, 1] -= d
    p[0, 1, 1] += d
    # We expect this error to be present in the result of the bundle adjustment
    expected_residuals = [ d, d ]
    expected_distances = [ 12.5, 12.5 ]

    # Run bundle adjustment
    P_estimated, residuals, distances = bundle_adjust(cams, p, full=True)

    # Check results
    assert np.max(np.abs((P_estimated - P))) < 1e-2
    assert np.max(np.abs((expected_residuals - residuals))) < 1e-2
    assert np.max(np.abs((expected_distances - distances))) < 1e-2



def test_upscaled():
    # Setup scene
    cam0 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(200, 0 ,0)))
    cam1 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(-200, 0 ,0)))
    cam2 = CameraModel(chip_size=(40, 30), focal_length=(50, 50),
        pose=Trafo3d(t=(0, 200 ,0)))
    cam3 = CameraModel(chip_size=(40, 30), focal_length=(40, 40),
        pose=Trafo3d(t=(0, -200, 0)))
    cams = [ cam0, cam1, cam2, cam3 ]
    n = 5000
    P = np.zeros((n, 3))
    P[:, 0] = np.random.uniform(-500, 500, n)
    P[:, 1] = np.random.uniform(-500, 500, n)
    P[:, 2] = np.random.uniform(500, 1500, n)
    #visualize_scene(cams, P)

    # Prepare points
    p = scene_to_chip(cams, P)

    # Disable some observations
    valid_mask = np.ones((n, len(cams)), dtype=bool)
    row_choices = np.random.choice(n, 100, replace=False)
    for r in row_choices:
        col_choices = np.random.choice(len(cams), 2, replace=True)
        valid_mask[r, col_choices] = False
    p[~valid_mask, :] = np.NaN

    P_estimated, residuals, distances = bundle_adjust(cams, p, full=True)

    # Check results
    absdiff = np.abs((P_estimated - P))
    assert np.max(absdiff) < 0.1

    assert np.max(residuals[valid_mask]) < 0.1
    assert np.all(np.isnan(residuals[~valid_mask]))

    assert np.max(distances[valid_mask]) < 0.1
    assert np.all(np.isnan(distances[~valid_mask]))
