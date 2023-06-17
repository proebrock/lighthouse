# Lighthouse: Large Scale Bundle Adjustment

So far when using bundle adjustment, we relied on multiple calibrated cameras with known poses. But many applications use a single freely moving camera to either reconstruct a 3D object or to track the camera movement. So let's try an example where we try to estimate 3D object points and camera poses at the same time.

This example uses some ideas from [this](https://scipy-cookbook.readthedocs.io/items/bundle_adjustment.html) SciPy tutorial which deals with large scale bundle adjustment problems. But we assume a single camera with known intrinsics instead of estimating the intrinsic parameters too.

## The problem

For this example we randomly distribute 40 individually colored spheres in space and use a single camera we move on a spiral-like trajectory around these spheres and take 40 images. The excessive movement provides rich information about our scene. Leaving the camera in a single area may make the problem unsolvable.

![](images/cam_trajectory.gif)

These 40 images are the only information we use: we not only reconstruct the 3D positions of the spheres but the camera poses as well!

To have a realistic setup, we must deal with the fact that not all spheres are visible in every image because of spheres located outside of the field of view or occlusion. We use spheres because in images it is easy to extract the center of the circle of the spheres' projection. In real world problems you would use a keypoint/feature extractor and correspondence matching to find corresponding 2D points in the images.

## Circle detection

First step is the detection of the circle centers. This is done by standard computer vision algorithms (Hough circle transformation). We sample the color of the circle at its center and use the color information for matching.

![](images/circle_detect.png)

Because we have a ground truth in this simulation we can check if the circles were detected properly. The grid shows circle detection errors for each camera image and for each 3D point. For white fields, there was no circle detection. Dark blue indicates a very low error. Other colors indicate presence of errors: out of cameras field of view, occlusion of other spheres or poor performance of circle detection algorithm.

![](images/circle_detect_errors.png)

All in all this is a realistic situation you would encounter in a real world scenario.

## Bundle adjustment

The bundle adjustment is done here using numerical optimization. The decision variable contains the 3D locations of the spheres (40*3=120 unknowns) and the 6D poses of the camera (40*6=240 unknowns), the rotation expresses as Rodrigues vector. So we have a 360 dimensional decision variable.

The starting points are very important to be able to calculate a solution for such a high dimensional problem. For the point locations we use an all-zero initialization, for the camera pose we use a single realistic camera pose of `Trafo3d(t=(0, 0, -1000))`. This seems to be enough in our example. In real world applications you would use e.g. the first two images and some assumptions to make a rough stereo-reconstruction of the scene to get good starting values.

Just like in traditional bundle adjustment, the objective function uses the optimizer-provided propositions for 3D points and camera poses and projects the 3D points onto the camera in the different poses. The resulting 2D chip points are subtracted from the real 2D chip points. We get a residuals for each 3D point and every view in X and Y.

The decision variable has 360 dimensions, the residuals vector has are (40*40*2=3200, minus the missing observations) dimensions. This means the Jacobian would have over 1.1 million entries (360*3200) and the numerical calculation would take some time. And since not every decision variable has influence on all residuals, we provide the optimizer with a sparse matrix that provides this information.

As we have seen in the results of the circle detection, we have to deal with outliers in our optimization. So we use stable optimization with a loss function to avoid skewed results.

## Ambiguity of the solution

After running the optimization, we can compare the ground truth (blue) with the result of the reconstruction (red). On first account the results seems to be pretty bad.

![](images/scene_raw_result.png)

But careful, we gave the optimizer the freedom to place the 3D points and camera poses arbitrarily just to match the provided images. So the result can be translated and rotated, just as it is visible here.

![](images/scene_transrot_comp.png)

After running a point registration with corresponding points on the 3D points we can compensate for translation and rotation. But the result is still different. The red points seems to be further distributed and the red camera positions are further out. This means we have a scaling problem! There are infinite solutions to our problem that are just different by a scaling factor. When compensating for that too, the reconstruction matches the ground truth.

![](images/scene_transrotscale_comp.png)

In real world scanning applications this problem exists too when just moving a single camera and this can only solved by additional assumptions. As soon as a second camera  (stereo configuration) is involved with a known distance between camera 1 and camera 2, the scaling problem is solved too.

## Quality of the solution

The residuals show the remaining errors in pixels. We see pretty low residuals. In places where we had detection errors of the circles the residuals are high. This means that the stable optimization is working.

![](images/residual_errors.png)

Comparing the results with the ground truth, we get and estimate of the quality of our solution. The point reconstruction errors are lower that 1.5mm.

![](images/point_errors.png)

The camera pose reconstruction errors are lower than 2.5mm (translational) and lower than 0.25 degrees (rotational).

![](images/pose_errors.png)



