# Lighthouse: Large Scale Bundle Adjustment

So far when using bundle adjustment, we relied on multiple calibrated cameras with known poses. But many applications use a single freely moving camera to either reconstruct a 3D object or to track the camera movement. So let's try an example where we try to estimate 3D object points and camera poses at the same time.

This example uses some ideas from [this](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) SciPy tutorial which deals with large scale bundle adjustment problems. But we assume a single camera with known intrinsics instead of estimating the intrinsic parameters too.

For this example we randomly distribute 40 individually colored spheres in space and use a single camera we move on a spiral-like trajectory around these spheres and take 40 images:

![](images/cam_trajectory.gif)

These 40 images are the only information we use: we not only reconstruct the 3D positions of the spheres (40*3=120 unknowns) but the camera poses as well (40*6=240 unknowns). To have a realistic setup, we must deal with the fact that not all spheres are visible in every image because of spheres located outside of the field of view or occlusion. We use spheres because in images it is easy to extract the center of the circle of the spheres' projection. In real world problems you would use a keypoint/feature extractor and correspondence matching to find corresponding 2D points in the images.

First step is the detection of the circle centers. This is done by standard computer vision algorithms (Hough circle transformation). We sample the color of the circle at its center and use the color information for matching.

![](images/circle_detect.png)

![](images/circle_detect_errors.png)

![](images/point_errors.png)

![](images/pose_errors.png)

![](images/residual_errors.png)

![](images/scene_raw_result.png)

![](images/scene_transrot_comp.png)

![](images/scene_transrotscale_comp.png)

