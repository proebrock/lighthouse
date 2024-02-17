# Lighthouse: Projector calibration

## The problem

Projectors can be used for all kind of cool computer vision applications like e.g. 3D reconstruction of objects with structured light. For some of the applications it is necessary to have a model of the projector. A projector can be modeled similar to a camera: while with the camera light is projected from the scene onto the camera chip, the projector projects points on the chip into the scene. Both use a lens or lens system, so both models have parameters like focal length, principal point, distortion parameters and an pose somewhere in a world coordinate system. This chapter is about identifying the model parameters of a given projector.

Calibrating cameras it's all about finding lots of pairs of 3D object points of the scene and 2D image points where the camera sees that object point. These object and image points can be fed into a calibration routine that estimates our model parameters. Main problem calibrating a projector is the fact that we cannot "see" the image projection of object points on the chip like with a camera.

There is a paper that describes an interesting approach here [[1]](#1). The paper does a projector calibration with a single camera, we extend this approach to a multi-view approach. Our setup has a projector in the center and cameras to the left and right of it. We place a Charuco calibration board in different poses in front of the setup and use the projector to project a couple of images on the calibration board that can be used to identify projector-camera pixel correspondences. The paper uses binary gray code patterns, we use phase shift encoding patterns (as described in the structured light chapter).

![](images/setup.png)

In this example we have 2 cameras and put that board into 12 poses. For phase shift correspondence matching we use 7 images each to identify the row and column, plus two pictures black (projector off) and white (projector all white). So during the calibration process we take 2 * 12 * (2 * 7 + 2) = 384 images.

## Determination of object and image points

On the left you see an image of a camera from the calibration board in a certain pose. A Charuco detection algorithm has detected all chessboard corners and one of these corners is marked with a blue plus sign. This image point has been detected with sub-pixel accuracy.

![](images/cam_projector_match.png)

The red dot in the camera image on the left is a circular patch of camera pixels in a radius of 10 pixels around the image point. Because we took images for phase shift correspondence matching, we can match those pixels to pixels on the projector. These pixels form a circle on the projector chip.

If we could somehow transfer the image point of the camera to the chip of the projector, then this would be the image point of the projector: where the object point of that particular Charuco board corner would be located if the projector was a camera. With multiple of these image points we could run a calibration for the projector in just the same way as we would for a camera. So how to we transfer this blue cross to the projector chip with loosing as little accuracy as possible?

To solve this problem we can make use of the fact that the calibration board is a 3D plane, therefore the relationship of the red points on the left and on the right is a [homography](https://en.wikipedia.org/wiki/Homography_(computer_vision)). We can use the OpenCV function `cv2.findHomography()` to estimate a homography matrix $H$ from the camera chip points and the matching projector chip points. With that matrix $H$ we can then translate the sub-pixel accurate camera image point to a sub-pixel accurate projector image point. The cool thing of just using some pixels in the region of the camera image point to calculate a local homography is that this way we can account for distortions of the camera and projector. The OpenCV function supports a stable estimation of the homography (e.g. RANSAC) to avoid wrong results in presence of some mismatches of camera and projector pixels.

In this example we have shown the procedure for a single camera, a single board pose and a single camera image point. If we do this for all, this is the result:

![](images/projector_image_points.png)

Each image shows the projector chip with the image points transformed using local homographies for a single board pose. First it is nice to see, that the points transformed from both of the cameras end up in the same area that a difference is hard to see. The top-left image was zoomed into a single point, so you can see the projector image points from two different cameras (red and green). The black point is the median of the projector image points from different cameras. These are the final projector image points.

## Calibration and results

The calibration itself is now pretty straight forward. We can feed the object and image point pairs of each camera and of the projector separately into an OpenCV routine like `cv2.calibrateCamera()`. This gives us intrinsics (focal length, principal points, distortion parameters) of each device. We also get the transformations from each device to each pose of the calibration board. From this we can estimate the pose of the cameras relative to the pose of the projector (which we choose as world coordinate system).

In a second step we run a single numerical optimization estimating all intrinsics and extrinsics by calculating differences of all image points and projected object points. Here are the results:

Focal lengths:
```
projector         cam0                 cam1
f=[540.  720.]    f=[1280   1280]      f=[1600   1600]   (model)
f=[572.3 773.4]   f=[1274.5 1275.1]    f=[1598.  1597.6] (initial guess)
f=[538.  715.4]   f=[1277.7 1277. ]    f=[1597.4 1597.2] (after optim.)
```

Principal points:
```
projector         cam0               cam1
c=[400.  300.]    c=[640.  400.]     c=[800.  600.]  (model)
c=[407.3 318.3]   c=[637.1 399.1]    c=[803.  600.1] (initial guess)
c=[396.9 295.4]   c=[638.7 400.2]    c=[801.2 600.8] (after optim.)
```

Projector distortion (for this experiment we did not use distortions on camera models and did not estimate distortions for cameras):
```
dist=[-0.05     0.1      0.1     -0.05     0.25]    (model)
dist=[-0.08336  0.56328  0.10902 -0.04836 -1.3373]  (initial guess)
dist=[-0.04993  0.12184  0.10194 -0.04806  0.15044] (after optim.)
```

Extrinsics cam0 (relative to projector, translation, rotation in RPY angles in degrees):
```
pose=([-200,    10,    0],   [ 3.,  16.,   1.])  (model)
pose=([-201. ,   6.3, 38.4], [ 4.2, 15.2,  1.2]) (initial guess)
pose=([-199.9,  10.,  -0.3], [ 2.6, 16.3,  0.9]) (after optim.)
```

Extrinsics cam1 (relative to projector, translation, rotation in RPY angles in degrees):
```
pose=([210,    -5,     3],   [  2.,  -14.,   -2.])  (model)
pose=([206.8,  -5.5,  43.4], [  3.5, -14.3,  -2.4]) (initial guess)
pose=([210.1,  -4.8,   0.3], [  1.6, -13.7,  -1.9]) (after optim.)
```

Finally we check the residuals after the initial guess of the model parameters and the final result after numerical optimization. The final result has a reprojection error of 0.3 pixels RMS which is okay.

![](images/residuals.png)

## References

<a id="1">[1]</a>
D. Moreno and G. Taubin, "Simple, Accurate, and Robust Projector-Camera Calibration," 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, Zurich, Switzerland, 2012, pp. 464-471, doi: 10.1109/3DIMPVT.2012.77.