# Lighthouse: Stereo Calibration

Calibration of two cameras in a stereo configuration.

## Scene and image generation

We setup two cameras in a stereo configuration

![](images/stereo_setup.png)

We vary the pose of the calibration board and snap images with both cameras.

## Solution

A possible solution we have already reviewed is the [multi camera calibration](../2d_calibrate_multiple) for the two cameras of our stereo configuration. This works flawlessly and could be used e.g. for comparison.

But OpenCV provides a function `cv::stereoCalibrate` to calibrate a stereo configuration of cameras. This does a global optimization over the intrinsics and extrinsics of both cameras and - in addition - provides the essential and fundamental matrices that can be used for geometric projection between both camera images.

Unfortunately there is no stereo calibration integrated in the ArUco package of OpenCV. So we use `cv2.aruco.detectMarkers` to detect the markers and image points, we generate the 3D object points ourselves from the parameters of the ChArUco board and finally use `cv::stereoCalibrate` to do the calibration.

Comparing the real extrinsics (top) - the transformation from the right camera to the left camera - with the estimated one (bottom), we can be happy with the result:

```
###### Camera pose ######
([247.,   3., -14.], [-1.5,  3. ,  2. ])
([247.6,   3.1, -14.2], [-1.8,  3. ,  2. ])
Error: dt=0.6, dr=0.26 deg
```

In addition we calculate the essential matrix E and fundamental matrix F by ourselves from the intrinsics and extrinsics of the cameras (see function `calculate_stereo_matrices()`) and compare it with the result of `cv::stereoCalibrate`. The function `calculate_stereo_matrices()` will be later used to calculate E and F for playing with projections between both camera images.
