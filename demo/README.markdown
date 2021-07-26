# Lighthouse: Demo

## Description

This application shows the capabilities of the *Lighthouse* framework.

First we setup the scene consisting of a red sphere embedded into a yellow plane. The camera view is slightly angled. The camera has some radial and tangential distortions.

![](images/scene.png)

Snapping the image lets the ray tracer intersect rays from every pixel on the camera chip with the mesh of the scene:

![](images/rays.png)

From the rays hitting the objects in the scene we can reconstruct color, shade and distance to the camera resulting in an RGB and depth image

![](images/images.png)

And of course the colored point cloud

![](images/pcl.png)

