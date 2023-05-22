# Lighthouse: Multi marker pose estimation

An application of 3D computer vision is the estimation of the pose of one or multiple objects observed by one or more cameras. We show how this can be done using markers.

## Scene and image generation

The object whose 6D pose we want to estimate is a yellow rectangle. We put ArUco markers with unique IDs in all four edges of the rectangle. We have a single fixed and calibrated 2D camera and we want to localize the plane in a single 2D image within the camera coordinate system.

![](images/scene.png)

## Solution

For the object with the markers on it the 3D positions of each marker corner in the object coordinate system in known. For our object these are located on a plane, but that is not necessary. These 3D points are the so-called *object points*. For 4 markers and 4 corners per marker, this makes 16 object points in total.

After taking an image from the object with its markers, we use a function of the `cv2.aruco` package to detect all markers and their corners in that image. These are the 2D *image points*. Because not all markers may be visible in the image, the number of image points may be lower than the number of object points. Here all points are visible.

For each detected image point we find the corresponding object point. Correspondences are denoted by lines between object and image point:

![](images/points.png)

Because we have a calibrated camera, we can use the camera model to project the object points onto the camera chip. The distances to the corresponding image points are a quality measure used in the numerical optimization. Here the projected object points are perfectly aligned with the image points:

![](images/projected_points.png)

The output shows the quality of the pose estimation:

```
residuals_rms:
    0.59
cam_to_object:
    ([-30,  20, 580], [-10., -20.,  20.])
cam_to_object estimated:
    ([-30.3,  19.8, 581.1], [ -9.9, -19.9,  20. ])
Difference: 1.19 mm, 0.15 deg
```

This is the plot of the residuals for each of the 16 object image point pairs and the differences in X and Y:

![](images/residuals.png)

The accuracy of the pose estimation could be increased by using more cameras.
