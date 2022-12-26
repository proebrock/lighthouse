# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import random as rand

import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d
from camsimlib.o3d_utils import mesh_transform
from . camera_model import CameraModel



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def test_look_at():
    camera_model = CameraModel((640, 480), focal_length=50)
    # Place in +X
    camera_model.place((100, 0, 0))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 1, 0], # e_x
        [0, 0, -1], # e_y
        [-1, 0, 0] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -X
    camera_model.place((-100, 0, 0))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [0, -1, 0], # e_x
        [0, 0, -1], # e_y
        [1, 0, 0] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Y
    camera_model.place((0, 100, 0))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, -1], # e_x
        [0, 0, -1], # e_y
        [0, -1, 0] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Y
    camera_model.place((0, -100, 0))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [0, 0, 1], # e_x
        [0, 0, -1], # e_y
        [0, 1, 0] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in +Z
    camera_model.place((0, 0, 100))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, -1, 0], # e_y
        [0, 0, -1] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)
    # Place in -Z
    camera_model.place((0, 0, -100))
    camera_model.look_at((0, 0, 0))
    r_expected = np.array([
        [1, 0, 0], # e_x
        [0, 1, 0], # e_y
        [0, 0, 1] # e_z
        ]).T
    r_actual = camera_model.get_pose().get_rotation_matrix()
    np.allclose(r_expected, r_actual)



def test_check_chip_edge_points():
    cam = CameraModel(chip_size=(40, 30), focal_length=(20, 10))
    distance = 10
    # Generate depth image with all corners pixels set to a certain distance,
    # all other pixels invalid
    depth_image = np.zeros((cam.get_chip_size()[1], cam.get_chip_size()[0]))
    depth_image[:] = np.NaN
    depth_image[0, 0] = distance
    depth_image[0, -1] = distance
    depth_image[-1, 0] = distance
    depth_image[-1, -1] = distance
    # Transform depth image resulting in 3D coordinates of those 4 pixels
    P1 = cam.depth_image_to_scene_points(depth_image)
    # Generate chip points from 0 to max pixels
    p = np.array([
        [ 0, 0, distance ],
        [ cam.get_chip_size()[0], 0, distance ],
        [ 0, cam.get_chip_size()[1], distance ],
        [ cam.get_chip_size()[0], cam.get_chip_size()[1], distance ],
        ])
    # Transform chip points resulting in 3D coordinates of those 4 pixels
    P2 = cam.chip_to_scene(p)
    # Compare!
    assert np.allclose(P1, P2)



def chip_to_scene_and_back(camera_model, rtol=1e-5, atol=1e-8):
    # Generate test points on chip
    width, height = camera_model.get_chip_size()
    focal_length = np.mean(camera_model.get_focal_length())
    min_distance = 0.01 * focal_length
    max_distance = 10 * focal_length
    num_points = 100
    p = np.hstack((
        (2.0 * width * np.random.rand(num_points, 1) - width) / 2.0,
        (2.0 * height * np.random.rand(num_points, 1) - height) / 2.0,
        min_distance + (max_distance-min_distance) * np.random.rand(num_points, 1)))
    # Transform to scene and back to chip
    P = camera_model.chip_to_scene(p)
    p2 = camera_model.scene_to_chip(P)
    # Should still be the same
    #print(np.nanmax(np.abs(p-p2)))
    assert np.allclose(p, p2, rtol=rtol, atol=atol)



def depth_image_to_scene_and_back(camera_model, rtol=1e-5, atol=1e-8):
    # Generate test depth image
    width, height = camera_model.get_chip_size()
    focal_length = np.mean(camera_model.get_focal_length())
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
    P = camera_model.depth_image_to_scene_points(img)
    img2 = camera_model.scene_points_to_depth_image(P)
    # Should still be the same (NaN at same places, otherwise numerically close)
    assert np.all(np.isnan(img) == np.isnan(img2))
    mask = ~np.isnan(img)
    #print(np.max(np.abs(img[mask]-img2[mask])))
    assert np.allclose(img[mask], img2[mask], rtol=rtol, atol=atol)



def test_roundtrips():
    # Simple configuration
    camera_model = CameraModel((640, 480), focal_length=50)
    chip_to_scene_and_back(camera_model)
    depth_image_to_scene_and_back(camera_model)
    # Two different focal lengths
    camera_model = CameraModel((800, 600), focal_length=(50, 60))
    chip_to_scene_and_back(camera_model)
    depth_image_to_scene_and_back(camera_model)
    # Principal point is off-center
    camera_model = CameraModel((600, 600), focal_length=1000,
                      principal_point=(250, 350))
    chip_to_scene_and_back(camera_model)
    depth_image_to_scene_and_back(camera_model)
    # Radial distortion
    camera_model = CameraModel((200, 200), focal_length=2400,
                      distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
    chip_to_scene_and_back(camera_model, atol=0.1)
    depth_image_to_scene_and_back(camera_model, atol=0.1)
    camera_model = CameraModel((100, 100), focal_length=4000,
                      distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
    chip_to_scene_and_back(camera_model, atol=0.1)
    depth_image_to_scene_and_back(camera_model, atol=0.1)
    # Transformations
    camera_model = CameraModel((100, 100), focal_length=200,
                      pose=Trafo3d(t=(0, 0, -500)))
    chip_to_scene_and_back(camera_model)
    depth_image_to_scene_and_back(camera_model)



def test_snap_empty_scene():
    # Get mesh object
    mesh = o3d.geometry.TriangleMesh()
    # Set up camera model and snap image
    cam = CameraModel((50, 50), 100, pose=Trafo3d(t=(0, 0, 500)))
    depth_image, color_image, pcl = cam.snap(mesh)
    # An empty image should result in all pixels being invalid and no scene points
    assert np.all(np.isnan(depth_image))
    assert np.all(np.isnan(color_image))
    assert np.asarray(pcl.points).size == 0
    assert np.asarray(pcl.colors).size == 0



def snap_knot(trafo_world_cam, trafo_world_object):
    mesh = o3d.io.read_triangle_mesh('data/knot.ply')
    if np.asarray(mesh.vertices).size == 0:
        raise Exception('Unable to load data file')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, trafo_world_object)
    cam = CameraModel((120, 90), 200, pose=trafo_world_cam)
    depth_image, color_image, pcl = cam.snap(mesh)
    return depth_image, color_image, pcl



def test_transform_object_and_cam():
    # Define camera position and object position and snap image
    trafo_world_cam = Trafo3d(t=(0, 0, -250))
    trafo_world_object = Trafo3d(t=(0, 0, 250), rpy=np.deg2rad([155, 25, 0]))
    depth_image1, color_image1, pcl1 = snap_knot(trafo_world_cam, trafo_world_object)
    # Move both the camera and the object by the same trafo T and snap image
    trafo = Trafo3d(t=(100, -1200, -40), rpy=np.deg2rad([-180, 90, 100]))
    trafo_world_cam = trafo * trafo_world_cam
    trafo_world_object = trafo * trafo_world_object
    depth_image2, color_image2, pcl2 = snap_knot(trafo_world_cam, trafo_world_object)
    # Both images should be the same and scene points vary by T
    assert np.nanmax(np.abs(depth_image1 - depth_image2)) < 1e-3
    assert np.nanmax(np.abs(color_image1 - color_image2)) < 1e-3
    assert np.max(np.abs(trafo * np.asarray(pcl1.points) - \
        np.asarray(pcl2.points))) < 1e-3
    assert np.max(np.abs(np.asarray(pcl1.colors) - \
        np.asarray(pcl2.colors))) < 1e-3



if __name__ == '__main__':
    pytest.main()
