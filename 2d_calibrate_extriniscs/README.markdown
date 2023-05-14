# Lighthouse: Multi Camera Extrinsics Calibration

Extrinsic calibration of multiple fixed cameras: Estimate the poses of all cameras in world coordinate system.

## Scene and image generation

We generate cube as a calibration object with Aruco markers on each face. This allows a calibration even of some cameras are facing each other:

![](images/calib_cube.png)

We use four fixed cameras. An we vary the pose of the calibration cube and snap images with all four cameras.

## Solution

For an initial guess we estimate the poses of the cameras relative to each other from the first image of all cameras. Then we estimate the poses of the markers relative to the first camera using all images of the first camera.

We set the world coordinate system to be the first camera. The main calibration uses the initial estimates for a numerical optimization over the poses of the other cameras relative to the first and the poses of the markers.

```
Running extrinsic calibration ...
    Done, residual RMS is 1.20
Comparing results ...
    Errors for cam0: 0.00 mm, 0.00 deg
    Errors for cam1: 2.69 mm, 0.53 deg
    Errors for cam2: 2.49 mm, 0.37 deg
    Errors for cam3: 2.63 mm, 0.57 deg
```

The result show sufficiently small deviations between the real camera positions and the estimated ones.
