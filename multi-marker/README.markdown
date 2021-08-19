# Lighthouse: Multi marker pose estimation

An application of 3D computer vision is the estimation of the pose of one or multiple objects observed by one or more cameras. We show how this can be done using markers.

## Scene and image generation

The object which 6D pose we want to estimate is a yellow rectangle. We put ArUco markers with unique IDs in all four edges of the rectangle. We have a single fixed and calibrated 2D camera and we want to localize the plane in a single 2D image within the camera coordinate system.

![](images/scene.png)

## Solution

For each marker the function `cv2.aruco.detectMarkers` returns the ID of the marker and the four marker corners. For 4 markers this makes 16 *image points* in 2D (one example shown in image):

![](images/img_points.png)

Because we defined the arrangement of the markers on the object, we have implicitly defined a set of *object points* in 3D (one example shown in image):

![](images/obj_points.png)

Our object points lie in a plane (with Z=0). But that is not necessary the case. You could e.g. put six markers on the sides of a six-sided dice to detect its pose and/or calibrate a multi-camera setup.

After determining the correspondences of the marker IDs of our object to the IDs found in the image, we can solve for the object pose from a number of *image point* to *object point* pairs!

This can be implemented as a numerical optimization, see `solvepnp`. Decision variable is the 6D pose of the object. The objective function projects the 3D object points to the camera chip and determines the distance to the 2D image points.

The result looks promising:

```
cam_to_object:
    ([-30.,  20., 580.], [170., -20.,  20.])
cam_to_object estimated:
    ([-29.7,  20.3, 580. ], [170., -20.,  20.])
Difference: 0.44 mm, 0.04 deg
```

And the per-point residuals measured in pixels look good too (subpixel accuracy):

![](images/residuals.png)

What happens when we cover three of the four markers? Well, this happens (you can simulate that in the code):

```
cam_to_object:
    ([-30.,  20., 580.], [170., -20.,  20.])
cam_to_object estimated:
    ([-29.6,  20. , 581.6], [170. , -19.9,  20.1])
Difference: 1.61 mm, 0.20 deg
```

Because of the loss of three of four markers, we only have the four corners of the single marker. This gives us very little of a lever to successfully identify the orientation of the object which is clearly visible in larger the errors.

The function doing the work (`solvepnp`) is implemented as a numerical optimization. Numerical optimizations need a good starting point. We can either just assume, that the object is somewhere in front of the camera. Or we can ask `cv2.aruco.estimatePoseSingleMarkers` to give us a guess of each marker position. It requires the camera model parameters as well as the marker square length as additional information. Then we can use the average pose of these estimates for each marker to get a very good starting point for the optimization in `solvepnp`.
