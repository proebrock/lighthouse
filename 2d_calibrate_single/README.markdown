# Lighthouse: Single Camera Calibration

The calibration of a single camera: The estimation of the camera intrinsic and extrinsic parameters.

## Scene and image generation

To calibrate a camera we use an [ChArUco](https://docs.opencv.org/master/df/d4a/tutorial_charuco_detection.html) board as calibration target and a camera. The camera must be placed in varying positions in the room looking at the calibration board in order to get enough information to solve for the intrinsic and extrinsic parameters of the camera.

We generate random camera positions here and a random position on the calibration target the camera is looking at. Then we move the camera in its view direction (Z-axis) until we get a maximized image of the calibration board.

This is done with a numerical optimization. Decision variable `x` is the displacement of the camera in Z direction. The objective function takes an image of a camera 3 times larger than the original and takes an image of the calibration board after moving it by `x` in Z direction. The result of the objective function is the number of pixels in the inner 3x3 field of the camera image (representing the original resolution of the camera) minus the pixels in the outer 8 fields of the camera (representing pixels that would not be shown in the original camera image because the camera is too far off).

## Solution

We use the OpenCV to detect the chessboard corners. The board in this example is not completely visible. You see the unique IDs of the chessboard corners thanks to the ArUco markers and the coordinate system of the calibration board:

![](images/markers.png)

Images with an insufficient number of corners are rejected.

Next we run the calibration. The resulting RMS of the projection error is displayed.

```
Reprojection error: 0.18 pixels
```

### Intrinsic camera parameters

The OpenCV calibration estimates the camera matrix as part of intrinsic camera parameters. Since we saved the camera model parameters in the JSON files when generating the images, we can compare the original model parameters with the estimated ones from OpenCV:

```
Camera matrix used in model
[[1500.    0.  600.]
 [   0. 1650.  450.]
 [   0.    0.    1.]]
Camera matrix as calibration result
[[1498.9    0.   602.5]
 [   0.  1648.7  449. ]
 [   0.     0.     1. ]]
Deviation of camera matrices
[[-1.1  0.   2.5]
 [ 0.  -1.3 -1. ]
 [ 0.   0.   0. ]]
```

And here the comparison of the distortion parameters:

```
Distortion coefficients used in model
[-0.8  0.8  0.   0.   0.   0.   0.   0.   0.   0.   0.   0. ]
Distortion coefficients as calibration result
[-0.798  0.771  0.    -0.001  0.166  0.     0.     0.     0.     0.
  0.     0.   ]
Deviation of distortion coefficients
[-0.002  0.029 -0.     0.001 -0.166  0.     0.     0.     0.     0.
  0.     0.   ]
```

The accuracy of the estimations depends besides other things on:

* The resolution of the generated camera parameters; the higher the resolution, the higher the accuracy
* The distortion model chosen for estimating the camera parameters; the above results are archived by running a vanilla OpenCV calibration. If we tell OpenCV that there are no distortion parameters k3, p1, p2, the estimations are better.

### Extrinsic camera parameters

The extrinsic camera parameters are the 6d poses of the cameras relative to the calibration target. And again we can compare the real and the estimated poses. The script calculates the mean translational and rotatory errors over all poses:

```
All trafos: dt=0.4, dr=0.09 deg
```

Here is a graphical representation of the camera poses relative to the calibration target:

![](images/extrinsics.png)
