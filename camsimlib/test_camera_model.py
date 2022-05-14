# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
import pytest
import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d
from camsimlib.o3d_utils import mesh_transform
from . camera_model import CameraModel



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



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



def test_snap_close_object():
    # Get mesh object
    mesh = o3d.io.read_triangle_mesh('data/triangle.ply')
    if np.asarray(mesh.vertices).size == 0:
        raise Exception('Unable to load data file')
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, Trafo3d(rpy=np.deg2rad([180, 0, 0])))
    # Set up camera model and snap image
    focal_length = 20
    pixels = 100
    distance = 5
    cam = CameraModel((pixels, pixels), focal_length,
                      pose=Trafo3d(t=(0, 0, -distance)))
    depth_image, _, _ = cam.snap(mesh)
    # Minimal distance in depth image is d in the middle of the image
    mindist = distance
    assert np.isclose(np.min(depth_image), mindist)
    # Maximum distance in depth image is at four corners of image
    xy_dist = ((pixels/2)*distance)/focal_length # Transform p/2 pixels to distance in scene
    maxdist = cam.scene_to_chip(np.array([[xy_dist, xy_dist, 0.0]]))[0, 2]
    assert np.isclose(np.max(depth_image), maxdist)



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
