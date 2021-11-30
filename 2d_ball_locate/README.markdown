# Lighthouse: 2D Ball Locate

This idea of this project is simple: We use a calibrated single 2D camera to estimate the 3D position of a ball of known radius. The camera image of the sphere of the ball is a circle. The center of the circle and the camera model give us a ray into the scene the sphere is located on. The radius of the circle helps us to estimate the distance of the sphere from the camera.

## Scene and image generation

The scene generation is pretty straight forward. We have a camera (left) and a sphere representing the ball.

![](images/scene.png)

The camera image shows the projection of the sphere onto the sensor chip resulting in a circle.

![](images/ball1.png)

Dependent on the camera model parameters the circle can be distorted.

## Solution

### First estimate

Using OpenCV we can do a simple thresholding (`cv2.threshold()`) to segment the image and to extract the contour of the circle (`cv2.findContours()`). With image moments (`cv2.moments()`) we can determine the center of gravity of the circle to estimate the circle center and the area to estimate the circle radius.

![](images/ball2.png)

Ignoring all but the simplest camera model parameters we can use the *focal length* and the [Intercept theorem](https://en.wikipedia.org/wiki/Intercept_theorem) to estimate the distance of the sphere from the camera:

```math
\frac{f}{r_c}=\frac{z}{r_s}\quad\Leftrightarrow\quad z=\frac{f\,r_s}{r_c}
```

### Improving estimation

![](images/ball3.png)

![](images/rays.png)