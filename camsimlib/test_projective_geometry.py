import pytest

import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d
from . projective_geometry import ProjectiveGeometry



class ProjectiveGeometryTest(ProjectiveGeometry):

    def get_chip_size(self):
        return np.array((400, 300))



def test_look_at():
    geometry = ProjectiveGeometryTest(focal_length=50)
    # Place in +X
    geometry.place((100, 0, 0))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 1, 0], # e_x
        [0, 0, -1], # e_y
        [-1, 0, 0] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -X
    geometry.place((-100, 0, 0))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, -1, 0], # e_x
        [0, 0, -1], # e_y
        [1, 0, 0] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Y
    geometry.place((0, 100, 0))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, -1], # e_x
        [0, 0, -1], # e_y
        [0, -1, 0] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Y
    geometry.place((0, -100, 0))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, 1], # e_x
        [0, 0, -1], # e_y
        [0, 1, 0] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Z
    geometry.place((0, 0, 100))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, -1, 0], # e_y
        [0, 0, -1] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Z
    geometry.place((0, 0, -100))
    geometry.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, 1, 0], # e_y
        [0, 0, 1] # e_z
        ]).T
    r_actual = geometry.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)



def test_check_chip_edge_points():
    geometry = ProjectiveGeometryTest(focal_length=(20, 10))
    distance = 10
    # Generate depth image with all corners pixels set to a certain distance,
    # all other pixels invalid
    depth_image = np.zeros((geometry.get_chip_size()[1], geometry.get_chip_size()[0]))
    depth_image[:] = np.NaN
    depth_image[0, 0] = distance
    depth_image[0, -1] = distance
    depth_image[-1, 0] = distance
    depth_image[-1, -1] = distance
    # Transform depth image resulting in 3D coordinates of those 4 pixels
    P1 = geometry.depth_image_to_scene_points(depth_image)
    # Generate chip points from 0 to max pixels
    p = np.array([
        [ 0, 0, distance ],
        [ geometry.get_chip_size()[0], 0, distance ],
        [ 0, geometry.get_chip_size()[1], distance ],
        [ geometry.get_chip_size()[0], geometry.get_chip_size()[1], distance ],
        ])
    # Transform chip points resulting in 3D coordinates of those 4 pixels
    P2 = geometry.chip_to_scene(p)
    # Compare!
    assert np.allclose(P1, P2)



def chip_to_scene_and_back(random_generator, geometry, rtol=1e-5, atol=1e-8):
    # Generate test points on chip
    width, height = geometry.get_chip_size()
    focal_length = np.mean(geometry.get_focal_length())
    min_distance = 0.01 * focal_length
    max_distance = 10 * focal_length
    num_points = 100
    p = np.hstack((
        random_generator.uniform(0.0, width, (num_points, 1)),
        random_generator.uniform(0.0, height, (num_points, 1)),
        random_generator.uniform(min_distance, max_distance, (num_points, 1))))
    # Transform to scene and back to chip
    P = geometry.chip_to_scene(p)
    p2 = geometry.scene_to_chip(P)
    # Should still be the same
    #print(np.nanmax(np.abs(p-p2)))
    assert np.allclose(p, p2, rtol=rtol, atol=atol)



def depth_image_to_scene_and_back(random_generator, geometry, rtol=1e-5, atol=1e-8):
    # Generate test depth image
    width, height = geometry.get_chip_size()
    focal_length = np.mean(geometry.get_focal_length())
    min_distance = 0.01 * focal_length
    max_distance = 10 * focal_length
    img = random_generator.uniform(min_distance, max_distance, (height, width))
    # Set up to every 10th pixel to NaN
    num_nan = (width * height) // 10
    nan_idx = np.hstack(( \
        random_generator.integers(0, width, size=(num_nan, 1)), \
        random_generator.integers(0, height, size=(num_nan, 1))))
    img[nan_idx[:, 1], nan_idx[:, 0]] = np.nan
    # Transform to scene and back
    P = geometry.depth_image_to_scene_points(img)
    img2 = geometry.scene_points_to_depth_image(P)
    # Should still be the same (NaN at same places, otherwise numerically close)
    assert np.all(np.isnan(img) == np.isnan(img2))
    mask = ~np.isnan(img)
    #print(np.max(np.abs(img[mask]-img2[mask])))
    assert np.allclose(img[mask], img2[mask], rtol=rtol, atol=atol)



def test_roundtrips(random_generator):
    # Simple configuration
    geometry = ProjectiveGeometryTest(focal_length=50)
    chip_to_scene_and_back(random_generator, geometry)
    depth_image_to_scene_and_back(random_generator, geometry)
    # Two different focal lengths
    geometry = ProjectiveGeometryTest(focal_length=(50, 60))
    chip_to_scene_and_back(random_generator, geometry)
    depth_image_to_scene_and_back(random_generator, geometry)
    # Principal point is off-center
    geometry = ProjectiveGeometryTest(focal_length=1000,
                      principal_point=(250, 350))
    chip_to_scene_and_back(random_generator, geometry)
    depth_image_to_scene_and_back(random_generator, geometry)
    # Radial distortion
    geometry = ProjectiveGeometryTest(focal_length=2400,
                      distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
    chip_to_scene_and_back(random_generator, geometry, atol=0.1)
    depth_image_to_scene_and_back(random_generator, geometry, atol=0.1)
    geometry = ProjectiveGeometryTest(focal_length=4000,
                      distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
    chip_to_scene_and_back(random_generator, geometry, atol=0.1)
    depth_image_to_scene_and_back(random_generator, geometry, atol=0.1)
    # Transformations
    geometry = ProjectiveGeometryTest(focal_length=200,
                      pose=Trafo3d(t=(0, 0, -500)))
    chip_to_scene_and_back(random_generator, geometry)
    depth_image_to_scene_and_back(random_generator, geometry)



def test_get_rays_principal_point():
    geometry = ProjectiveGeometryTest()
    points = np.array((
        geometry.get_principal_point(),
        ))
    rays = geometry.get_rays(points)
    assert np.all(np.isclose(rays.origs, (0, 0, 0)))
    assert np.all(np.isclose(rays.dirs,  (0, 0, 1)))
