# Lighthouse: Multi Camera Calibration

Calibration of multiple cameras. Especially interesting to determine the 6d poses of multiple cameras in a world coordinate system.

## Scene and image generation

We use four fixed cameras with different poses, resolutions and focus lengths:

![](images/four_cams.png)

We vary the pose of the calibration board and snap images with all four cameras.

## Solution

The script calibrates all four cameras separately, resulting in the camera intrinsics as well as the camera extrinsics: the transformations from the calibration board to each of the cameras for each pose of the calibration board. We can calculate the transformations from camera 0 to each of the other cameras for each calibration board pose and for the final result average over all these.

We express all camera poses relative to camera 0 and then average over all estimated poses for each camera. For each camera we compare the nominal camera pose from the image generation (first line) with the estimated camera pose from the calibration (second line) and determine the translational and rotatory difference between both:

```
###### Camera poses ######
 ------------- cam0 -------------
([0., 0., 0.], [0., 0., 0.])
([ 0., -0.,  0.], [0., 0., 0.])
Error: dt=0.0, dr=0.00 deg
 ------------- cam1 -------------
([-678.6, -916.1, -748.5], [ 36. ,  -3.1, 153.2])
([-681. , -919.9, -755. ], [ 36. ,  -3.2, 153.1])
Error: dt=7.9, dr=0.06 deg
 ------------- cam2 -------------
([ 621. , -563.1,  199.6], [  52.9,  -12.7, -119.6])
([ 618.4, -562.2,  200. ], [  52.7,  -12.6, -119.6])
Error: dt=2.7, dr=0.19 deg
 ------------- cam3 -------------
([ -89.6, -812.2,  567.6], [ 66.6,  60.5, 155.4])
([ -88.3, -819.7,  565.4], [ 67.1,  60.6, 155.6])
Error: dt=7.9, dr=0.27 deg
```

The solution presented here does separate calibrations for each camera and later averages over the resulting transformations to the get the best guess. Better would be a global optimization over the intrinsics and extrinsics of all cameras in one go, using the previous solution as initial guess!
