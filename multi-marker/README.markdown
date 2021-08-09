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

This can be implemented as a numerical optimization. Decision variable is the 6D pose of the object. The objective function projects the 3D object points to the camera chip and determines the distance to the 2D image points.

The result looks promising:

```
Detecting markers ...
    4 markers found in image
cam_T_plane:
    ([-148. ,   21.8,  577.2], [171.7,  11.2,  28.2])
cam_T_plane estimated:
    ([-147.9,   21.9,  577.3], [171.7,  11.2,  28.2])
Difference: 0.22 mm, 0.02 deg
```

And the per-point residuals measured in pixels look good too (subpixel accuracy):

![](images/residuals.png)

What happens when we cover three of the four markers? Well, this happens:

```
Detecting markers ...
    1 markers found in image
cam_T_plane:
    ([-148. ,   21.8,  577.2], [171.7,  11.2,  28.2])
cam_T_plane estimated:
    ([-147.9,   22.1,  577.9], [171.9,  11.3,  28.1])
Difference: 0.78 mm, 0.21 deg
```
Because of the loss of four markers and the limited resolution of the image we are less successfully able to estimate position and angle of the object, clearly visible in larger the error.
