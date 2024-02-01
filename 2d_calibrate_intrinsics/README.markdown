# Lighthouse: Single Camera Intrinsics Calibration

The calibration of a single camera: The estimation of the camera intrinsic parameters. For the minimal pin-hole camera model this is the focus length(s) and the principal point (intersection point of camera chip with optical axis). But may contain distortion parameters to model lens distortions.

## Scene and image generation

To calibrate a camera we use an chessboard as calibration target and a camera. The camera must be placed in varying positions relative to the calibration board in order to get enough information to solve for the intrinsic parameters of the camera.

We simply generate a camera object with known model parameters, generate the poses of the board relative to the camera and take a number of images.

## Solution

We use OpenCV to detect the chessboard corners in each image. These are the so called *image points*, 2D image coordinates. For each of the image points we can find corresponding *object points*, 3D world coordinates of the same corners in the 3D chessboard. Because the board is flat, the Z coordinates of these points is zero.

The calibration now tries to find one set of intrinsic camera parameters and transformation for each pose of the board so that the 3D object points projected into the images are as close as possible to the image points. The root mean square (RMS) error of those on-chip pixel distances is called *reprojection error*.

```
Running intrinsics calibration ...
    Reprojection error is 0.16
Comparing results ...
    Orig focal lengths [1500 1650]
    Estm focal lengths [1488.  1636.3]
    Orig principal points [600. 450.]
    Estm principal points [595.7 452.5]
    Orig distortion [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    Estm distortion [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```
