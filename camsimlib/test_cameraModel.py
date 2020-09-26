# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
import random as rand
from trafolib.trafo3d import Trafo3d
from . mesh_object import MeshObject
from . camera_model import CameraModel



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def chip_to_scene_and_back(cam, rtol=1e-5, atol=1e-8):
    # Generate test points on chip
    width, height = cam.get_pixel_size()
    f = np.mean(cam.get_focus_length())
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
    width, height = cam.get_pixel_size()
    f = np.mean(cam.get_focus_length())
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
    cam = CameraModel((640, 480), f=50)
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Two different focal lengths
    cam = CameraModel((800, 600), f=(50,60))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Principal point is off-center
    cam = CameraModel((600, 600), f=1000, c=(250,350))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)
    # Radial distortion
    cam = CameraModel((200, 200), f=2400, distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
    chip_to_scene_and_back(cam, atol=0.1)
    depth_image_to_scene_and_back(cam, atol=0.1)
    cam = CameraModel((100, 100), f=4000, distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
    chip_to_scene_and_back(cam, atol=0.1)
    depth_image_to_scene_and_back(cam, atol=0.1)
    # Transformations
    cam = CameraModel((100, 100), f=200, trafo=Trafo3d(t=(0,0,-500)))
    chip_to_scene_and_back(cam)
    depth_image_to_scene_and_back(cam)



def test_snap_empty_scene():
    # Get mesh object
    mesh = MeshObject()
    # Set up camera model and snap image
    cam = CameraModel((50,50), 100, trafo=Trafo3d(t=(0,0,500)), shading_mode='flat')
    dImg, cImg, P = cam.snap(mesh)
    # An empty image should result in all pixels being invalid and no scene points
    assert(np.all(np.isnan(dImg)))
    assert(np.all(np.isnan(cImg)))
    assert(P.size == 0)



def test__snap_close_object():
    # Get mesh object
    mesh = MeshObject()
    mesh.load('data/triangle.ply')
    mesh.demean()
    mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
    # Set up camera model and snap image
    f = 20
    p = 100
    d = 5
    cam = CameraModel((p,p), f, trafo=Trafo3d(t=(0,0,-d)), shading_mode='flat')
    dImg, cImg, P = cam.snap(mesh)
    # Minimal distance in depth image is d in the middle of the image
    mindist = d
    assert(np.isclose(np.min(dImg), mindist))
    # Maximum distance in depth image is at four corners of image
    xy = ((p/2)*d)/f # Transform p/2 pixels to distance in scene
    maxdist = cam.scene_to_chip(np.array([[xy, xy, 0.0]]))[0, 2]
    assert(np.isclose(np.max(dImg), maxdist))



def test_snap_triangle():
    # Get mesh object
    mesh = MeshObject()
    mesh.load('data/triangle.ply')
    mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
    # Set up camera model and snap image
    d = 500 # Distance object/camera
    l = 100 # Length of triangle
    pix = np.array([120,100])
    f = np.array([150,200])
    cam = CameraModel(pix, f, trafo=Trafo3d(t=(0,0,-d)), shading_mode='flat')
    dImg, cImg, P = cam.snap(mesh)
    # Valid/invalid pixels should be same in dImg and cImg
    assert(np.array_equal( \
            np.isnan(dImg),
            np.isnan(cImg[:,:,0]),
            ))
    # Get indices of valid image points
    cImg[np.isnan(cImg)] = 0 # we look for white pixels, so set NaN pixels to 0
    idx = np.where(cImg > 0)
    mm = f*l/d # Side length of triangle in x and y
    # Check dimensions in image against simplified camera model
    assert(np.allclose( \
        # Minimum of valid pixel coordinates
        np.min(idx,axis=1)[0:2],
        # Left and top of image due to camera model
        np.array([pix[1]/2 - mm[1], pix[0]/2]),
        atol=1.0
        ))
    assert(np.allclose( \
        # Maximum of valid pixel coordinates
        np.max(idx,axis=1)[0:2],
        # Right and buttom of image due to camera model
        np.array([pix[1]/2, pix[0]/2 + mm[0]]),
        atol=1.0
        ))
    # Check scene points against mesh vertices
    assert(np.allclose( \
        # Minimum of coordinates of scene points
        np.min(P, axis=0),
        # Minimum of coordinates of vertices from mesh
        np.min(mesh.vertices, axis=0),
        atol=1.0
        ))
    assert(np.allclose( \
        # Maximum of coordinates of scene points
        np.max(P, axis=0),
        # Maximum of coordinates of vertices from mesh
        np.max(mesh.vertices, axis=0),
        atol=1.0
        ))



def snap_knot(T_world_cam, T_world_object):
    mesh = MeshObject()
    mesh.load('data/knot.ply')
    mesh.demean()
    mesh.transform(T_world_object)
    cam = CameraModel((120, 90), 200, trafo=T_world_cam)
    dImg, cImg, P = cam.snap(mesh)
    return dImg, cImg, P



def test_transform_object_and_cam():
    # Define camera position and object position and snap image
    T_world_cam = Trafo3d(t=(0, 0, -250))
    T_world_object = Trafo3d(t=(0, 0, 250), rpy=np.deg2rad([155, 25, 0]))
    dImg1, cImg1, P1 = snap_knot(T_world_cam, T_world_object)
    # Move both the camera and the object by the same trafo T and snap image
    T = Trafo3d(t=(100, -1200, -40), rpy=np.deg2rad([-180, 90, 100]))
    T_world_cam = T * T_world_cam
    T_world_object = T * T_world_object
    dImg2, cImg2, P2 = snap_knot(T_world_cam, T_world_object)
    # Both images should be the same and scene points vary by T
    assert(np.isclose(np.nanmax(np.abs(dImg1 - dImg2)), 0))
    assert(np.isclose(np.nanmax(np.abs(cImg1 - cImg2)), 0))
    assert(np.isclose(np.max(np.abs((T * P1) - P2)), 0))



if __name__ == '__main__':
    pytest.main()

