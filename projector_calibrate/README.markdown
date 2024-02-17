
Have a look at [[1]](#1).

![](images/setup.png)

![](images/cam_projector_match.png)

![](images/projector_image_points.png)

![](images/residuals.png)

Focal lengths
```
projector         cam0                 cam1
f=[540. 720.]     f=[1280 1280]        f=[1600 1600]
f=[572.3 773.4]   f=[1274.5 1275.1]    f=[1598.  1597.6]
f=[538.  715.4]   f=[1277.7 1277. ]    f=[1597.4 1597.2]
```

Principal point
```
projector         cam0               cam1
c=[400. 300.]     c=[640. 400.]      c=[800. 600.]
c=[407.3 318.3]   c=[637.1 399.1]    c=[803.  600.1]
c=[396.9 295.4]   c=[638.7 400.2]    c=[801.2 600.8]
```

Projector distortion
```
dist=[-0.05     0.1      0.1     -0.05     0.25]
dist=[-0.08336  0.56328  0.10902 -0.04836 -1.3373]
dist=[-0.04993  0.12184  0.10194 -0.04806  0.15044]
```

Extrinsics cam0
```
pose=([-200,    10,    0],   [ 3.,  16.,   1.])
pose=([-201. ,   6.3, 38.4], [ 4.2, 15.2,  1.2])
pose=([-199.9,  10.,  -0.3], [ 2.6, 16.3,  0.9])
```

Extrinsics cam1
```
pose=([210,    -5,     3],   [  2.,  -14.,   -2.])
pose=([206.8,  -5.5,  43.4], [  3.5, -14.3,  -2.4])
pose=([210.1,  -4.8,   0.3], [  1.6, -13.7,  -1.9])
```




## References

<a id="1">[1]</a>
D. Moreno and G. Taubin, "Simple, Accurate, and Robust Projector-Camera Calibration," 2012 Second International Conference on 3D Imaging, Modeling, Processing, Visualization & Transmission, Zurich, Switzerland, 2012, pp. 464-471, doi: 10.1109/3DIMPVT.2012.77.