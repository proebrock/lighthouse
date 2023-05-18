# Lighthouse: Stereo Calibration

Calibration of two cameras in a stereo configuration.

## Scene and image generation

We setup two cameras in a stereo configuration

![](images/stereo_setup.png)

We vary the pose of the calibration board and snap images with both cameras.

## Solution

OpenCV provides a function `cv::stereoCalibrate` to calibrate a stereo configuration of cameras. This does a global optimization over the intrinsics and extrinsics of both cameras and - in addition - provides the essential and fundamental matrices that can be used for geometric projection between both camera images.

```
Running stereo calibration ...
    Reprojection error is 0.22
Left camera
    Real focal lengths [1500. 1500.]
    Estm focal lengths [1501.5 1501.5]
    Real principal points [600. 450.]
    Estm principal points [600.8 451.5]
    Orig distortion [ 0.1 -0.1  0.   0.   0.   0.   0.   0.   0.   0.   0.   0. ]
    Estm distortion [ 0.104 -0.114  0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.   ]
Right camera
    Real focal lengths [1500. 1350.]
    Estm focal lengths [1498.7 1348.4]
    Real principal points [600. 450.]
    Estm principal points [598.1 451.8]
    Real distortion [-0.1  0.1  0.   0.   0.   0.   0.   0.   0.   0.   0.   0. ]
    Estm distortion [-0.095  0.059  0.     0.     0.     0.     0.     0.     0.     0.
  0.     0.   ]
Camera pose cam_right_to_cam_left
    Real ([-247.3,    5.6,    1.2], [ 1.6, -2.9, -2.1])
    Estm ([-247.2,    5.9,   -1. ], [ 1.6, -2.8, -2.1])
    Errors: 2.25 mm, 0.10 deg
```

In addition we calculate the **essential matrix** $`E`$ and the **fundamental matrix** $`F`$ by ourselves from the intrinsics and extrinsics of the cameras (see function `calculate_stereo_matrices()`) and compare it with the result of `cv::stereoCalibrate`. The function `calculate_stereo_matrices()` will be later used to calculate $`E`$ and $`F`$ for playing with projections between both camera images.
