# Bundle adjustment

Camera models project a 3D point P(X,Y,Z) from the scene to a point p(u,v) on the chip of the camera. This projection is not reversible: All 3D points of a scene that are projected to the same 2D point on the chip lie on a ray from the optical center of the camera through this point on the chip. So for reversing the projection we need additional information. This can be camera-internal: A Time-of-Flight camera provides a distance to a point on the chip that makes it possible to reconstruct the 3D point in the scene. Or we can use more than one camera to reconstruct points in the scene.

So multiple cameras observing the same point in the scene result in one ray from each camera intersecting in the3D point in the scene. This bundle of rays has led to the term [Bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment) which is determining the best estimate for a 3d point in the scene from observations of multiple cameras.

![](images/bundle_adjust.png)

The example shown here uses four cameras to observe a sphere. The projected image of all four spheres are circles. If we determine the center of the circle of each camera image, we have a 3D point in the scene all cameras observe and we have four 2D coordinates under which this point is seen by all the cameras. This is the perfect scenario to test the bundle adjustment.

## Bundle adjustment as an optimization problem

Let's sketch a solution using numerical optimization!

Unknown is the point in the scene in 3D. So we have three unknowns in our **decision variable**.

As **initial values** we can use the fact, that the point is in front of the camera. We could start e.g. at [0, 0, 100].

The **objective function** takes the decision variables as parameter and additionally the four 2d coordinates of the circle centers and the four camera objects. Using the camera objects, we can project the estimated sphere position in `x` to the chip of each camera.

The projection can be done using `scene_to_chip` and providing `x` (reshaped to (1, 3)). The function provides the point on the chip p(u,v) along with the distance (since the camsimlib simulates a RGB/ToF camera). This last component we can ignore. Alternatively we can use `cv2.projectPoints` to execute the camera model.

Now we have all the 2d points where the sphere would appear in the camera image if it was at `x`and the real positions of the projected sphere in the images. The difference is the result of the objective function!

For the **optimizer** we use `least_squares` from `scipy.optimize`.

The expected result for the sphere position is [ 47 -61 -76].

