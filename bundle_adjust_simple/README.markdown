# Lighthouse: Bundle adjustment

Camera models project a 3D point P(X,Y,Z) from the scene to a point p(u,v) on the chip of the camera. This projection is not reversible: All 3D points of a scene that are projected to the same 2D point on the chip lie on a ray from the optical center of the camera through this point on the chip. So for reversing the projection we need additional information. This can be camera-internal: A Time-of-Flight camera provides a distance to a point on the chip that makes it possible to reconstruct the 3D point in the scene. Or we can use more than one camera to reconstruct points in the scene.

So multiple cameras observing the same point in the scene result in one ray from each camera intersecting in the 3D point in the scene. This bundle of rays has led to the term [Bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) which is determining the best estimate for a 3d point in the scene from observations of multiple cameras.

## Scene and image generation

We use four cameras all observing a single sphere:

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
Estimated sphere center at [ 47.07122771 -61.25024237 -76.26689746] mm
Error [ 0.07122771 -0.25024237 -0.26689746] mm
Residuals per cam [0.23 0.17 0.15 0.11] pix
Errors per cam [0.18 0.12 0.12 0.07] mm
```

And here the visualization of the bundle adjustment result:

![](images/bundle_adjust.png)

### Displaced camera

```
Re-running bundle adjustment after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 51.61614745 -57.93188616 -78.17095287] mm
Error [ 4.61614745  3.06811384 -2.17095287] mm
Residuals per cam [ 6.98 15.21  7.13  4.98] pix
Errors per cam [ 5.6  10.42  5.42  3.26] mm
```

```
Re-running SAC bundle adjustment after misaligning camera 1...
Real sphere center at [ 47 -61 -76] mm
Estimated sphere center at [ 47.14953064 -61.2599172  -76.36904521] mm
Error [ 0.14953064 -0.2599172  -0.36904521] mm
Residuals per cam [ 0.16 23.47  0.09  0.11] pix
Errors per cam [ 0.13 16.06  0.07  0.07] mm
Inlier cameras: [ True False  True  True]
```







