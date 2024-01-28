import pytest

import numpy as np
import open3d as o3d
import cv2

from trafolib.trafo3d import Trafo3d
from . camera_model import CameraModel



def project_points_camera_model_and_opencv(chip_size, focal_length,
    principal_point, distortion, pose, object_points):
    cameraMatrix = np.array([
        [ focal_length[0], 0.0, principal_point[0] ],
        [ 0.0, focal_length[1], principal_point[1] ],
        [ 0.0, 0.0, 1.0 ],
    ])
    # Convention for pose for CameraModel: world to cam
    # Convention for tvec, rvec in OpenCV: cam to world
    tvec = pose.inverse().get_translation()
    rvec = pose.inverse().get_rotation_rodrigues()
    # Create camera object and project object points to chip
    cam = CameraModel(chip_size=chip_size, focal_length=focal_length,
        principal_point=principal_point, distortion=distortion,
        pose=pose)
    image_points_cam = cam.scene_to_chip(object_points)
    image_points_cam = image_points_cam[:, 0:2] # Skip distances
    # Use OpenCV to project object points to chip
    image_points_cv2, _ = cv2.projectPoints(object_points, rvec, tvec, cameraMatrix, distortion)
    image_points_cv2 = image_points_cv2.reshape((-1, 2))
    return image_points_cam, image_points_cv2



def test_camera_model_opencv_compare_simple():
    # Intrinsics
    chip_size = np.array([32, 20])
    focal_length = np.array([50.0, 50.0])
    principal_point = 0.5 * chip_size
    distortion = np.zeros(4)
    # Extrinsics
    pose = Trafo3d()
    # Object points (3D points in scene)
    object_points = np.array([
        [0.0, 0.0, 100.0],
        [30.0, 0.0, 100.0],
        [31.0, 0.0, 100.0],
        [32.0, 0.0, 100.0],
        [33.0, 0.0, 100.0],
        [0.0, -18.0, 100.0],
        [0.0, -19.0, 100.0],
        [0.0, -20.0, 100.0],
        [0.0, -21.0, 100.0],
    ])
    image_points_cam, image_points_cv2 = project_points_camera_model_and_opencv( \
        chip_size, focal_length, principal_point, distortion, pose, object_points)
    assert np.all(np.isclose(image_points_cam, image_points_cv2))



def test_camera_model_opencv_compare_enhanced():
    # Intrinsics
    chip_size = np.array([32, 20])
    focal_length = np.array([50.0, 45.0])
    principal_point = np.array([15, 11])
    distortion = np.array((-0.1, 0.1, 0.05, -0.05, 0.2))
    # Extrinsics
    pose = Trafo3d(t=(10, -20, 30), rodr=(0.1, 0.5, -0.2))
    # Object points (3D points in scene)
    object_points = np.array([
        [0.0, 0.0, 100.0],
        [30.0, 0.0, 100.0],
        [31.0, 0.0, 100.0],
        [32.0, 0.0, 100.0],
        [33.0, 0.0, 100.0],
        [0.0, -18.0, 100.0],
        [0.0, -19.0, 100.0],
        [0.0, -20.0, 100.0],
        [0.0, -21.0, 100.0],
    ])
    image_points_cam, image_points_cv2 = project_points_camera_model_and_opencv( \
        chip_size, focal_length, principal_point, distortion, pose, object_points)
    assert np.all(np.isclose(image_points_cam, image_points_cv2))



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
    mesh.transform(trafo_world_object.get_homogeneous_matrix())
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

