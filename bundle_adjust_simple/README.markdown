# Lighthouse: Bundle adjustment

Camera models project a 3D point P(X,Y,Z) from the scene to a 2D point p(u,v) on the chip of the camera. This projection is not reversible: All 3D points of a scene that are projected to the same 2D point on the chip lie on a ray from the optical center of the camera through this point on the chip. So for reversing the projection we need additional information. This can be camera-internal: A Time-of-Flight camera provides a distance to a point on the chip that makes it possible to reconstruct the 3D point in the scene. Or we can use more than one camera to reconstruct points in the scene.

So multiple cameras observing the same point in the scene result in one ray from each camera intersecting in the 3D point in the scene. This bundle of rays has led to the term [Bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) which is determining the best estimate for a 3d point in the scene from observations of multiple cameras.

## Scene and image generation

We use four cameras all observing a single sphere. The poses of the cameras in the world coordinate system are known (for example they have been determined with a multi camera calibration):

![](images/scene.png)

The projected image of the sphere on the camera chips are circles (or distorted circles dependent on the camera distortion). We can determine the center (of gravity) of the circles on the camera chips to get 2D coordinates for each camera. With this information we can try to reconstruct the 3D position of the sphere using bundle adjustment.

![](images/circles.png)

## Solution

### 2D

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

### Misaligned camera

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

### Misaligned camera and sample consensus

The previous result is encouraging, we can detect errors and estimate the quality of our calculations. But it would be nice to be able identify the faulty camera in the scenario above and make a better estimation based on all the good cameras?

To do this, we borrow the idea from the [RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus) approach. But instead of using random samples of cameras to identify our model, we generate all combinations of samples.

The minimal number of cameras to do a bundle adjustment is two. So we generate all combinations of two cameras out of four, which makes 6 combinations (for 2 cameras out of `n` these are `(n-1)*n/2` combinations):

```
 [[0 1]
 [0 2]
 [0 3]
 [1 2]
 [1 3]
 [2 3]]
```

Next we solve the bundle adjustment for all these combinations and get the reprojection errors for each camera:

```
 [[ 8.53951024  7.42929375 19.62060131 23.50094534]
 [ 0.11058322 23.42217499  0.11320118  0.19967323]
 [ 0.14254241 23.54234729  0.16044673  0.10307853]
 [10.73332668 10.31403186 11.45110528  8.75600226]
 [22.21751923  6.28119474 20.0178321   6.10341072]
 [ 0.22497886 23.43788233  0.05881517  0.0797954 ]]
```

Then we use a threshold (e.g. 3 pixels) to determine acceptable reprojection errors and to determine solutions with the maximum numbers of cameras agreeing on this solution (the consensus):

```
 [[False False False False]
 [ True False  True  True]
 [ True False  True  True]
 [False False False False]
 [False False False False]
 [ True False  True  True]]
```

One of the solutions with the most inliers is solution 1 (second line). We run the final bundle adjustment with cameras 0, 2 and 3 with this solution:

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

# 3D bundle adjustment

