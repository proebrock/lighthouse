# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
import random as rand
import open3d as o3d
from trafolib.trafo3d import Trafo3d
from . camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def chip_to_scene_and_back(cam, rtol=1e-5, atol=1e-8):
    # Generate test points on chip
    width, height = cam.get_chip_size()
    f = np.mean(cam.get_focal_length())
    min_distance = 0.01 * f
    max_distance = 10 * f
    n = 100
    p = np.hstack((
        (2.0 * width * np.random.rand(n,1) - width) / 2.0,
        (2.0 * height * np.random.rand(n,1) - height) / 2.0,
        min_distance + (max_distance-min_distance) * np.random.rand(n,1)))
    # Transform to scene and back to chip
    P = cam.chip_to_scene(p)
    p2 = cam.scene_to_chip(P)
    # Should still be the same
    #print(np.nanmax(np.abs(p-p2)))
    assert(np.allclose(p, p2, rtol=rtol, atol=atol))



def depth_image_to_scene_and_back(cam, rtol=1e-5, atol=1e-8):
    # Generate test depth image
    width, height = cam.get_chip_size()
    f = np.mean(cam.get_focal_length())
    min_distance = 0.01 * f
    max_distance = 10 * f
    img = min_distance + (max_distance-min_distance) * np.random.rand(height, width)
    # Set up to every 10th pixel to NaN
    num_nan = (width * height) // 10
    nan_idx = np.hstack(( \
        np.random.randint(0, width, size=(num_nan, 1)), \
        np.random.randint(0, height, size=(num_nan, 1))))
    img[nan_idx[:,1], nan_idx[:,0]] = np.nan
    # Transform to scene and back
    P = cam.depth_image_to_scene_points(img)
    img2 = cam.scene_points_to_depth_image(P)
    # Should still be the same (NaN at same places, otherwise numerically close)
    assert np.all(np.isnan(img) == np.isnan(img2))
    mask = ~np.isnan(img)
    #print(np.max(np.abs(img[mask]-img2[mask])))
    assert np.allclose(img[mask], img2[mask], rtol=rtol, atol=atol)



def test__roundtrips():
    # Simple configuration
    cam = CameraModel((640, 480), focal_length=50)
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Two different focal lengths
    cam = CameraModel((800, 600), focal_length=(50,60))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Principal point is off-center
    cam = CameraModel((600, 600), focal_length=1000, principal_point=(250,350))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Radial distortion
    cam = CameraModel((200, 200), focal_length=2400, distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
    chip_to_scene_and_back(cam, atol=0.1)
    depth_image_to_scene_and_back(cam, atol=0.1)
    cam = CameraModel((100, 100), focal_length=4000, distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
    chip_to_scene_and_back(cam, atol=0.1)
    depth_image_to_scene_and_back(cam, atol=0.1)
    # Transformations
    cam = CameraModel((100, 100), focal_length=200, camera_pose=Trafo3d(t=(0,0,-500)))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)



def test_snap_empty_scene():
    # Get mesh object
    mesh = o3d.geometry.TriangleMesh()
    # Set up camera model and snap image
    cam = CameraModel((50,50), 100, camera_pose=Trafo3d(t=(0,0,500)), shading_mode='flat')
    depth_image, color_image, pcl = cam.snap(mesh)
    # An empty image should result in all pixels being invalid and no scene points
    assert(np.all(np.isnan(depth_image)))
    assert(np.all(np.isnan(color_image)))
    assert(np.asarray(pcl.points).size == 0)
    assert(np.asarray(pcl.colors).size == 0)



def test__snap_close_object():
    # Get mesh object
    mesh = o3d.io.read_triangle_mesh('data/triangle.ply')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, Trafo3d(rpy=np.deg2rad([180,0,0])))
    # Set up camera model and snap image
    f = 20
    p = 100
    d = 5
    cam = CameraModel((p,p), f, camera_pose=Trafo3d(t=(0,0,-d)), shading_mode='flat')
    depth_image, color_image, pcl = cam.snap(mesh)
    # Minimal distance in depth image is d in the middle of the image
    mindist = d
    assert(np.isclose(np.min(depth_image), mindist))
    # Maximum distance in depth image is at four corners of image
    xy = ((p/2)*d)/f # Transform p/2 pixels to distance in scene
    maxdist = cam.scene_to_chip(np.array([[xy, xy, 0.0]]))[0, 2]
    assert(np.isclose(np.max(depth_image), maxdist))



def snap_knot(T_world_cam, T_world_object):
    mesh = o3d.io.read_triangle_mesh('data/triangle.ply')
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, T_world_object)
    cam = CameraModel((120, 90), 200, camera_pose=T_world_cam)
    depth_image, color_image, pcl = cam.snap(mesh)
    return depth_image, color_image, pcl



def test_transform_object_and_cam():
    # Define camera position and object position and snap image
    T_world_cam = Trafo3d(t=(0, 0, -250))
    T_world_object = Trafo3d(t=(0, 0, 250), rpy=np.deg2rad([155, 25, 0]))
    depth_image1, color_image1, pcl1 = snap_knot(T_world_cam, T_world_object)
    # Move both the camera and the object by the same trafo T and snap image
    T = Trafo3d(t=(100, -1200, -40), rpy=np.deg2rad([-180, 90, 100]))
    T_world_cam = T * T_world_cam
    T_world_object = T * T_world_object
    depth_image2, color_image2, pcl2 = snap_knot(T_world_cam, T_world_object)
    # Both images should be the same and scene points vary by T
    assert(np.isclose(np.nanmax(np.abs(depth_image1 - depth_image2)), 0))
    assert(np.isclose(np.nanmax(np.abs(color_image1 - color_image2)), 0))
    assert(np.allclose(T * np.asarray(pcl1.points),
                       np.asarray(pcl2.points)))
    assert(np.allclose(np.asarray(pcl1.colors),
                       np.asarray(pcl2.colors)))



if __name__ == '__main__':
    pytest.main()

