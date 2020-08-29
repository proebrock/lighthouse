# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np
import random as rand
from trafolib.Trafo3d import Trafo3d
from CameraModel import CameraModel



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def chipToSceneAndBack(cam, rtol=1e-5, atol=1e-8):
	# Generate test points on chip
	width, height = cam.getPixelSize()
	f = np.mean(cam.getFocusLength())
	min_distance = 0.01 * f
	max_distance = 10 * f
	n = 100
	p = np.hstack((
		(2.0 * width * np.random.rand(n,1) - width) / 2.0,
		(2.0 * height * np.random.rand(n,1) - height) / 2.0,
		min_distance + (max_distance-min_distance) * np.random.rand(n,1)))
	# Transform to scene and back to chip
	P = cam.chipToScene(p)
	p2 = cam.sceneToChip(P)
	# Should still be the same
	#print(np.nanmax(np.abs(p-p2)))
	assert(np.allclose(p, p2, rtol=rtol, atol=atol))



def depthImageToSceneAndBack(cam, rtol=1e-5, atol=1e-8):
	# Generate test depth image
	width, height = cam.getPixelSize()
	f = np.mean(cam.getFocusLength())
	min_distance = 0.01 * f
	max_distance = 10 * f
	img = min_distance + (max_distance-min_distance) * np.random.rand(width, height)
	# Set up to every 10th pixel to NaN
	num_nan = (width * height) // 10
	nan_idx = np.hstack(( \
		np.random.randint(0, width, size=(num_nan, 1)), \
		np.random.randint(0, height, size=(num_nan, 1))))
	img[nan_idx[:,0], nan_idx[:,1]] = np.nan
	# Transform to scene and back
	P = cam.depthImageToScenePoints(img)
	img2 = cam.scenePointsToDepthImage(P)
	# Should still be the same (NaN at same places, otherwise numerically close)
	assert np.all(np.isnan(img) == np.isnan(img2))
	mask = ~np.isnan(img)
	#print(np.max(np.abs(img[mask]-img2[mask])))
	assert np.allclose(img[mask], img2[mask], rtol=rtol, atol=atol)



def test_RoundTrips():
	# Simple configuration
	cam = CameraModel((640, 480), f=50)
	chipToSceneAndBack(cam)
	depthImageToSceneAndBack(cam)
	# Two different focal lengths
	cam = CameraModel((800, 600), f=(50,60))
	chipToSceneAndBack(cam)
	depthImageToSceneAndBack(cam)
	# Principal point is off-center
	cam = CameraModel((600, 600), f=1000, c=(250,350))
	chipToSceneAndBack(cam)
	depthImageToSceneAndBack(cam)
	# Radial distortion
	cam = CameraModel((200, 200), f=2400, distortion=(0.02, -0.16, 0.0, 0.0, 0.56))
	chipToSceneAndBack(cam, atol=0.1)
	depthImageToSceneAndBack(cam, atol=0.1)
	cam = CameraModel((100, 100), f=4000, distortion=(-0.5, 0.3, 0.0, 0.0, -0.12))
	chipToSceneAndBack(cam, atol=0.1)
	depthImageToSceneAndBack(cam, atol=0.1)
    # Transformations
	cam = CameraModel((100, 100), f=200, T=Trafo3d(t=(0,0,-500)))
	chipToSceneAndBack(cam)
	depthImageToSceneAndBack(cam)



if __name__ == '__main__':
	pytest.main()

