# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
import pytest
import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d
from . projective_geometry import ProjectiveGeometry



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def test_look_at():
    projective_geometry = ProjectiveGeometry((640, 480), focal_length=50)
    # Place in +X
    projective_geometry.place((100, 0, 0))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 1, 0], # e_x
        [0, 0, -1], # e_y
        [-1, 0, 0] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -X
    projective_geometry.place((-100, 0, 0))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, -1, 0], # e_x
        [0, 0, -1], # e_y
        [1, 0, 0] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Y
    projective_geometry.place((0, 100, 0))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, -1], # e_x
        [0, 0, -1], # e_y
        [0, -1, 0] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Y
    projective_geometry.place((0, -100, 0))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, 1], # e_x
        [0, 0, -1], # e_y
        [0, 1, 0] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Z
    projective_geometry.place((0, 0, 100))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, -1, 0], # e_y
        [0, 0, -1] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Z
    projective_geometry.place((0, 0, -100))
    projective_geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, 1, 0], # e_y
        [0, 0, 1] # e_z
        ]).T
    r_actual = projective_geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)



def chip_to_scene_and_back(projective_geometry, rtol=1e-5, atol=1e-8):
    # Generate test points on chip
    width, height = projective_geometry.get_chip_size()
    focal_length = np.mean(projective_geometry.get_focal_length())
    min_distance = 0.01 * focal_length
    max_distance = 10 * focal_length
    num_points = 100
    p = np.hstack((
        (2.0 * width * np.random.rand(num_points, 1) - width) / 2.0,
        (2.0 * height * np.random.rand(num_points, 1) - height) / 2.0,
        min_distance + (max_distance-min_distance) * np.random.rand(num_points, 1)))
    # Transform to scene and back to chip
    P = projective_geometry.chip_to_scene(p)
    p2 = projective_geometry.scene_to_chip(P)
    # Should still be the same
    #print(np.nanmax(np.abs(p-p2)))
    assert np.allclose(p, p2, rtol=rtol, atol=atol)



def depth_image_to_scene_and_back(projective_geometry, rtol=1e-5, atol=1e-8):
    # Generate test depth image
    width, height = projective_geometry.get_chip_size()
    focal_length = np.mean(projective_geometry.get_focal_length())
    min_distance = 0.01 * focal_length
    max_distance = 10 * focal_length
    img = min_distance + (max_distance-min_distance) * np.random.rand(height, width)
    # Set up to every 10th pixel to NaN
    num_nan = (width * height) // 10
    nan_idx = np.hstack(( \
        np.random.randint(0, width, size=(num_nan, 1)), \
        np.random.randint(0, height, size=(num_nan, 1))))
    img[nan_idx[:, 1], nan_idx[:, 0]] = np.nan
    # Transform to scene and back
    P = projective_geometry.depth_image_to_scene_points(img)
    img2 = projective_geometry.scene_points_to_depth_image(P)
    # Should still be the same (NaN at same places, otherwise numerically close)
    assert np.all(np.isnan(img) == np.isnan(img2))
    mask = ~np.isnan(img)
    #print(np.max(np.abs(img[mask]-img2[mask])))
    assert np.allclose(img[mask], img2[mask], rtol=rtol, atol=atol)



def test__roundtrips():
    # Simple configuration
    projective_geometry = ProjectiveGeometry((640, 480), focal_length=50)
    chip_to_scene_and_back(projective_geometry)
    depth_image_to_scene_and_back(projective_geometry)
    # Two different focal lengths
    projective_geometry = ProjectiveGeometry((800, 600), focal_length=(50, 60))
    chip_to_scene_and_back(projective_geometry)
    depth_image_to_scene_and_back(projective_geometry)
    # Principal point is off-center
    projective_geometry = ProjectiveGeometry((600, 600), focal_length=1000,
                      principal_point=(250, 350))
    chip_to_scene_and_back(projective_geometry)
    depth_image_to_scene_and_back(projective_geometry)
    # Radial distortion
    projective_geometry = ProjectiveGeometry((200, 200), focal_length=2400,
                      distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
    chip_to_scene_and_back(projective_geometry, atol=0.1)
    depth_image_to_scene_and_back(projective_geometry, atol=0.1)
    projective_geometry = ProjectiveGeometry((100, 100), focal_length=4000,
                      distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
    chip_to_scene_and_back(projective_geometry, atol=0.1)
    depth_image_to_scene_and_back(projective_geometry, atol=0.1)
    # Transformations
    projective_geometry = ProjectiveGeometry((100, 100), focal_length=200,
                      pose=Trafo3d(t=(0, 0, -500)))
    chip_to_scene_and_back(projective_geometry)
    depth_image_to_scene_and_back(projective_geometry)



if __name__ == '__main__':
    pytest.main()
