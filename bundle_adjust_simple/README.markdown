# Lighthouse: Bundle adjustment

Camera models project a 3D point P(X,Y,Z) from the scene to a 2D point p(u,v) on the chip of the camera. This projection is not reversible: All 3D points of a scene that are projected to the same 2D point on the chip lie on a ray from the optical center of the camera through this point on the chip. So for reversing the projection we need additional information. This can be camera-internal: A time-of-flight camera provides a distance to a point on the chip that makes it possible to reconstruct the 3D point in the scene. Or we can use more than one camera to reconstruct points in the scene.

So multiple cameras observing the same point in the scene result in one ray from each camera intersecting in the 3D point in the scene. This bundle of rays has led to the term [Bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) which is determining the best estimate for a 3d point in the scene from observations of multiple cameras.

## Scene and image generation

We use four cameras all observing a single sphere. The poses of the cameras in the world coordinate system are known (for example they have been determined with a multi camera calibration):

![](images/scene.png)

The projected image of the sphere on the camera chips are circles (or distorted circles dependent on the camera distortion). We can determine the center (of gravity) of the circles on the camera chips to get 2D coordinates for each camera. With this information we can try to reconstruct the 3D position of the sphere using bundle adjustment.

![](images/circles.png)

## Solution

### 2D bundle adjustment

Let's sketch a solution using numerical optimization!

Unknown is the point in the scene in 3D. So we have three unknowns in our **decision variable**.

As **initial values** we can use the fact, that the point is in front of the camera. We could start e.g. at [0, 0, 100].

The **objective function** takes the decision variables as parameter and additionally the four 2d coordinates of the circle centers and the four camera objects. Using the camera objects, we can project the estimated sphere position in `x` to the chip of each camera.

The projection can be done using `scene_to_chip` and providing `x` (reshaped to (1, 3)). The function provides the point on the chip p(u,v) along with the distance (since the camsimlib simulates a RGB/ToF camera). This last component (the distance) we ignore here since we assume to have a 2D camera. Alternatively we can use `cv2.projectPoints` to execute the camera model.

Now we have all the 2D points where the sphere would appear in the camera image if it was at `x`and the real positions of the projected sphere in the images. The difference is the result of the objective function!

For the **optimizer** we use `least_squares` from `scipy.optimize`.

The expected result for the sphere position is [ 47 -61 -76].

Here is the program output:

```
Running bundle adjustment ...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 47.1 -61.3 -76.3] mm
Error [ 0.1 -0.3 -0.3] mm
Reprojection error per cam [0.2 0.2 0.2 0.1] pix
Errors per cam [0.2 0.1 0.1 0.1] mm
```

We see the real and estimated position of the sphere and the difference between both. For the reprojection errors we project the final estimated position of the sphere to the camera to each camera and determine the distance to the circle center of this camera. Because the reprojection error is measured in pixels, it is hard to estimate the error in the scene. And the error we got by subtracting real and estimated positions of the sphere are only available because we know the real sphere position. A solution is to determine the distance between the estimated sphere position and the ray of each camera (last line). This gives a good estimate of the error in the scene.

And here the visualization of the bundle adjustment result:

![](images/bundle_adjust.png)

### 2D bundle adjustment and misaligned camera

But why use four cameras if only two cameras would be sufficient? Because with this redundancy of more than 2 cameras we get information about the quality of our estimation and we get some tolerance against misconfiguration of the system, for example a misaligned camera. Especially in industrial applications this possibility to identify misaligned cameras is very important.

Lets use our previous configuration but give camera 1 a good kick which rotates the camera by 1 degree around the y axis of the camera coordinate system. The result looks like this:

![](images/bundle_adjust_misaligned.png)

Running the previous bundle adjustment directly shows the problem. Reprojection error and errors per camera went up.

```
Re-running bundle adjustment after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 51.6 -57.9 -78.2] mm
Error [ 4.6  3.1 -2.2] mm
Reprojection error per cam [ 7.  15.2  7.1  5. ] pix
Errors per cam [ 5.6 10.4  5.4  3.3] mm
```

The errors are a little bit higher for camera 1 (second value in the vectors). But the numerical optimization tries to minimize the least squares of the error, so the error is distributed over all cameras.

### 2D bundle adjustment, misaligned camera and sample consensus

The previous result is encouraging, we can detect errors and estimate the quality of our calculations. But it would be nice to be able identify the faulty camera in the scenario above and make a better estimation based on all the good cameras!?

To do this, we borrow the idea from the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) approach. But instead of using random samples of cameras to identify our model, we generate all combinations of cameras to do a minimal bundle adjustment.

The minimal number of cameras to do a bundle adjustment is two. So we generate all combinations of two cameras out of four, which makes 6 combinations (for 2 cameras out of `n` these are `(n-1)*n/2` combinations):

```
 [[0 1]
 [0 2]
 [0 3]
 [1 2]
 [1 3]
 [2 3]]
```

Next we solve the bundle adjustment for all these six combinations and get the reprojection errors for each of the four cameras:

```
 [[ 8.5  7.4 19.6 23.5]
 [ 0.1 23.4  0.1  0.2]
 [ 0.1 23.5  0.2  0.1]
 [10.7 10.3 11.5  8.8]
 [22.2  6.3 20.   6.1]
 [ 0.2 23.4  0.1  0.1]]
```

Then we use a threshold of 3 pixels to determine acceptable reprojection errors and to determine solutions with the maximum numbers of cameras agreeing on this solution (the consensus):

```
 [[False False False False]
 [ True False  True  True]
 [ True False  True  True]
 [False False False False]
 [False False False False]
 [ True False  True  True]]
```

One of the solutions with the most inliers is solution 1 (second line) which was generated with cameras 0 and 2 (and no camera 1). We run the final bundle adjustment with cameras 0, 2 and 3 with this solution:

```
Re-running SAC bundle adjustment (threshold=3.0 pix) after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 47.1 -61.3 -76.4] mm
Error [ 0.1 -0.3 -0.4] mm
Reprojection error per cam [ 0.2 23.5  0.1  0.1] pix
Errors per cam [ 0.1 16.1  0.1  0.1] mm
Inlier cameras: [ True False  True  True]
```

This is exactly what we were looking for. We identified the misaligned camera. And we got a near perfect estimate of the sphere position with all the remaining cameras.

### 3D bundle adjustment

Lets assume we have not a 2D camera but a 3D camera, for example a time-of-flight camera. A camera like this provides us with a depth image or a point cloud of the scene. To reconstruct the position of the sphere using a point cloud, a single 3D camera would suffice. We use four cameras and we know the camera poses in the world coordinate system. So we can transform the point clouds of all cameras into a single coordinate system. The result looks like this (color encodes the camera the point was taken from):

![](images/point_cloud.png)

We assume we know the radius of the sphere in the scene. So we can start a numerical optimization fitting a sphere of known radius into the point cloud:

```
Running sphere fitting ...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 47.  -61.  -75.9] mm
Error [-0.   0.   0.1] mm
Residuals per cam [0. 0. 0. 0.] mm
```
The residuals are the RMS distances of the points from each camera to the final sphere.

### 3D bundle adjustment and misaligned camera

Again we can try to check the robustness of the system against misaligned cameras. And again we intentionally misalign camera 1 by rotating it by 1 degree around the y axis of the camera coordinate system.

The resulting combined point cloud of all cameras looks like expected:

![](images/point_cloud_misaligned.png)


Running the 3D bundle adjustment like before yields this result:

```
Re-running sphere fitting after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 49.9 -59.  -76.3] mm
Error [ 2.9  2.  -0.3] mm
Residuals per cam [2.2 6.1 1.7 2. ] mm
```

Just like for the 2D case we can see the reduced accuracy of our estimation.

### 3D bundle adjustment, misaligned camera and sample consensus

Again we try to improve our solution by identifying the misaligned camera and to compute a better solution with the remaining camera. And again we use a sample consensus approach.

We run a 3D bundle adjustment on the points of each of the 4 cameras separately and determine the residuals per camera for all cameras:

```
 [[ 0.1  7.9  0.1  0.1]
 [ 8.5  0.1  8.3 10.6]
 [ 0.1  7.9  0.1  0.1]
 [ 0.1  7.9  0.1  0.1]]
```

Next we use a threshold of 1mm to determine the camera inliers per solution
Then we use a threshold of 1 millimeter to determine acceptable residual errors and to determine solutions with the maximum numbers of cameras agreeing on this solution (the consensus):

```
 [[ True False  True  True]
 [False  True False False]
 [ True False  True  True]
 [ True False  True  True]]
```

One of the solutions with the most inliers is solution 0 (first line) which was generated with cameras 0 (and no camera 1). We run the final bundle adjustment with cameras 0, 2 and 3 with this solution:

```
Re-running SAC sphere fitting (threshold=1.0 mm) after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 47.  -61.  -75.9] mm
Error [ 0.  -0.   0.1] mm
Residuals per cam [0.1 7.9 0.1 0.1] mm
Inlier cameras: [ True False  True  True]
```

This is exactly what we were looking for. We identified the misaligned camera. And we got a near perfect estimate of the sphere position with all the remaining cameras.