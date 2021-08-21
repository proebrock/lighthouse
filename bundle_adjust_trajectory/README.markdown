# Lighthouse: Reconstruct trajectory

After we implemented a simple bundle adjustment example, we put the implementation to use to reconstruct the trajectory of a flying target.

## Scene and image generation

We set up a scene of four fixed cameras, all calibrated and referenced to a world coordinate system. The cameras observe a ball being thrown into the air. It starts at position `s0` with a speed vector `v0`. The ball is affected by gravity (acceleration `[0, 0, 9.81] m/s^2`) and eventually returns back to earth (X/Y plane). The result is a proper ballistic curve. The curve is sampled every `dt` seconds by taking images of the scene with all cameras.

```
s0 = [ 200 -300 -150] mm
v0 = [-10  20 100] mm/s
|v0| = 102.47 mm/s
dt = 0.50 s
times = [0.00, 0.50, .., 20.50] s
number of samples = 42
```

The resulting scene looks like this:

![](images/scene.png)

## Solution

The solution is pretty forward. At each point in time we check the images of the four cameras for the centers of the circle which are the projection of the sphere to that particular camera chip. With at least 2 circle centers we can run a bundle adjustment to locate the sphere in the room. Doing that with every time step, we can reconstruct the trajectory of the sphere.

The bundle adjustment is based on numerical optimization. Numerical optimization needs a starting point. One way to get a starting point for the next time step is to use the position of the sphere at the current time step, because with a dense sampling, the sphere cannot be that far away. Or we could model the movement of the sphere having a position, a speed and an acceleration and to use this model to calculate an estimate for its location at the next time step.

We can check the result by comparing the real and the estimated positions of the sphere. Or we can check the error estimate of the bundle adjustment because the redundancy of cameras allows us this error estimate.

![](images/errors.png)

In this example the circle has been removed from frame #11 of camera 1. The reconstruction is still possible with the remaining cameras.