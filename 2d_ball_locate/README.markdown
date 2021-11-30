# Lighthouse: 2D Ball Locate

This idea of this project is simple: We use a calibrated single 2D camera to estimate the 3D position of a ball of known radius. The camera image of the sphere of the ball is a circle. The center of the circle and the camera model give us a ray into the scene the sphere is located on. The radius of the circle helps us to estimate the distance of the sphere from the camera.

## Scene and image generation

The scene generation is pretty straight forward. We have a camera (left) and a sphere representing the ball.

![](images/scene.png)

The camera image shows the projection of the sphere onto the sensor chip resulting in a circle.

![](images/ball1.png)

Dependent on the camera model parameters the circle can be distorted.

We generate around 100 images with the sphere at different positions.

## Solution

### First estimate

Using OpenCV we can do a simple thresholding (`cv2.threshold()`) to segment the image and to extract the contour of the circle (`cv2.findContours()`). With image moments (`cv2.moments()`) we can determine the center of gravity of the circle to estimate the circle center and the area to estimate the circle radius.

![](images/ball2.png)

This estimate of circle center and radius is not perfect, since the circle can be distorted, as seen in this image.

Ignoring all but the simplest camera model parameters we can use the *focal length* and the [Intercept theorem](https://en.wikipedia.org/wiki/Intercept_theorem) to estimate the distance of the sphere from the camera. Let $f$ be the focal length, $r_c$ the circle radius, $r_s$ the sphere radius and $z$ the estimated distance of the sphere from the camera:

```math
\frac{f}{r_c}=\frac{z}{r_s}\quad\Leftrightarrow\quad z=\frac{f\cdot r_s}{r_c}
```

This is the error distribution for all randomly chosen sphere positions. Since we expect that the estimation of the correct sphere position will be harder the greater the distance from the camera, we plot the real z distance of the sphere on the first axis. In RGB colors we see the deviations of estimated minus real sphere position for the single coordinates and the absolute distance in cyan.

![](images/error1.png)

The estimate gets worse the further the sphere is away. And we seem to systematically underestimate the distance of the sphere (Z coordinate) from the camera: the blue points are all below zero. Overall the estimate is okay but not great: A total distance of 3000mm and absolute errors of roughly 300mm is about 10 percent.

### Improving estimation

To improve the estimate of the sphere position from the previous step, we can use the contour points of the circle (red), that have been extracted using `cv2.findContours()`.

![](images/ball3.png)

Each of these points contour points describes a ray into the scene. To improve the circle estimate, we can fit the circle into this cone-shaped ray bundle.

![](images/rays.png)

A simple way to do this is to run a numerical optimization to minimize the distances of the sphere to all these rays by varying the estimated sphere center.

The error distribution after the optimization is different than before. X and Y errors seem to be random and increase with the distance of the sphere from the camera. We see a systematical error in underestimating Z. The absolute error is much better than the absolute error of the initial estimate.

![](images/error2.png)

**TODO**: Understand systematical errors in Z to improve estimation of sphere distance.