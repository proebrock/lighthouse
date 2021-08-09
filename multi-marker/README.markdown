# Lighthouse: Multi marker pose estimation

An application of 3D computer vision is the estimation of the pose of one or multiple objects observed by one or more cameras. We show how this can be done using markers.

## Scene and image generation

The object which 6D pose we want to estimate is a yellow rectangle. We put ArUco markers in all four edges of the rectangle. We have a single fixed and calibrated 2D camera and we want to localize the plane in a single 2D image within the camera coordinate system.

![](images/scene.png)

## Solution

![](images/marker.png)


```
Detecting markers ...
    4 markers found in image
cam_T_plane:
    ([-148. ,   21.8,  577.2], [171.7,  11.2,  28.2])
cam_T_plane estimated:
    ([-147.9,   21.9,  577.3], [171.7,  11.2,  28.2])
Difference: 0.22 mm, 0.02 deg
```


![](images/residuals.png)
