# TODO

## Next steps

* Test case for checking coordinate system orientations: snap images, move object/camera and see how the object moves in the image (OpenCV matchTemplate)
* Better visualization of results in open3d:
    * Objects, cameras, coordinate systems
	* Camera field of view, frustum
	* Rays, intersection with triangle
	* Point cloud
* Parallelize loop over all rays in CameraModel::snap using Python multiprocessing
* Investigate problem with vertex normals in open3d, check implementation in open3d
* Save scratchapixel.com
* Enable/disable culling in triangle-ray intersection

## Further ideas

* Model tangential distortion; check numerical solutions for radial/tangential distortions (forward+backward)
* Run hand-eye calibration examples in simulator
* Run bundle adjustment examples in simulator
* Model 2d laser triangulation sensor
* Model projector for active lighting simulations
* Add noise model to camera model

