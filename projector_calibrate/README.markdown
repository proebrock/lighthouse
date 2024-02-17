# Lighthouse: Projector calibration

## The problem

Projectors can be used for all kind of cool computer vision applications like e.g. 3D reconstruction of objects with structured light. For some of the applications it is necessary to have a model of the projector. A projector can be modeled similar to a camera: while with the camera light is projected from the scene onto the camera chip, the projector projects points on the chip into the scene. Both use a lens or lens system, so both models have parameters like focal length, principal point, distortion parameters and an pose somewhere in a world coordinate system. This chapter is about identifying the model parameters of a given projector.

Calibrating cameras it's all about finding lots of pairs of 3D object points of the scene and 2D image points where the camera sees that object point. These object and image points can be fed into a calibration routine that estimates our model parameters. Main problem calibrating a projector is the fact that we cannot "see" the image projection of object points on the chip like with a camera.

There is a paper that describes an interesting approach here [[1]](#1). The paper does a projector calibration with a single camera, we extend this approach to a multi-view approach. Our setup has a projector in the center and cameras to the left and right of it. We place a Charuco calibration board in different poses in front of the setup and use the projector to project a couple of images on the calibration board that can be used to identify projector-camera pixel correspondences. The paper uses binary gray code patterns, we use phase shift encoding patterns (as described in the structured light chapter).

![](images/setup.png)

We have 2 cameras and put that board into 12 poses. For phase shift correspondence matching we use 7 images each to identify the row and column, plus two pictures black (projector off) and white (projector all white). So during the calibration process we take 2 * 12 * (2 * 7 + 2) = 384 images.

## Determination of object and image points

![](images/cam_projector_match.png)

![](images/projector_image_points.png)

Focal lengths
```
projector         cam0                 cam1
f=[540.  720.]    f=[1280   1280]      f=[1600   1600]   (model)
f=[572.3 773.4]   f=[1274.5 1275.1]    f=[1598.  1597.6] (initial guess)
f=[538.  715.4]   f=[1277.7 1277. ]    f=[1597.4 1597.2] (after optim.)
```

Principal point
```
projector         cam0               cam1
c=[400.  300.]    c=[640.  400.]     c=[800.  600.]  (model)
c=[407.3 318.3]   c=[637.1 399.1]    c=[803.  600.1] (initial guess)
c=[396.9 295.4]   c=[638.7 400.2]    c=[801.2 600.8] (after optim.)
```

Projector distortion
```
dist=[-0.05     0.1      0.1     -0.05     0.25]    (model)
dist=[-0.08336  0.56328  0.10902 -0.04836 -1.3373]  (initial guess)
dist=[-0.04993  0.12184  0.10194 -0.04806  0.15044] (after optim.)
```

Extrinsics cam0
```
pose=([-200,    10,    0],   [ 3.,  16.,   1.])  (model)
pose=([-201. ,   6.3, 38.4], [ 4.2, 15.2,  1.2]) (initial guess)
pose=([-199.9,  10.,  -0.3], [ 2.6, 16.3,  0.9]) (after optim.)
```

Extrinsics cam1
```
pose=([210,    -5,     3],   [  2.,  -14.,   -2.])  (model)
pose=([206.8,  -5.5,  43.4], [  3.5, -14.3,  -2.4]) (initial guess)
pose=([210.1,  -4.8,   0.3], [  1.6, -13.7,  -1.9]) (after optim.)
```

![](images/residuals.png)



## References

<a id="1">[1]</a>
D. Moreno and G. Taubin, "Simple, Accurate, and Robust Projector-Camera Calibration," 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, Zurich, Switzerland, 2012, pp. 464-471, doi: 10.1109/3DIMPVT.2012.77.