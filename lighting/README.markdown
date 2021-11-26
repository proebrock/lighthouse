# Lighthouse: Lighting

The current ray tracer supports a single light source in the scene. If not configured differently, we assume a point light source at the same position as the camera. This makes ray tracing easy: If a camera ray intersects with the mesh at a certain point, this point is lighted by the light source as well.

Now there is support for two types of light sources:

* **Point light source** at a freely configurable point (`lighting_mode='point'`)
* **Parallel light** of a configurable direction (`lighting_mode='parallel'`)

This image illustrates the effects of different lighting modes. In the top row of images the lighting mode is `'point'`. In the middle image, the light source is at the camera position and then it is moved to the left (negative X, left image) and right (positive X, right image). This causes the expected shadow effects.

The bottom row of images shows the effect of the lighting mode `'parallel'`. The direction of the light in the middle image is the view direction of the camera (positive Z axis). It is then modified to 45 degrees upwards (negative Y, left image) and to 45 degrees downwards (positive Y, right image). With the expected shadow casting.

![](images/knot.png)

So far we cannot really distinguish the difference between a point light source and parallel lighting. Lets take an image of a plane. (The plane consists of a certain number of triangles, not just two triangles.) Again the top row shows the lighting mode `'point'` with a light source moved left and right. The brightness of a point is determined by the angle between the normal vector of the surface (constant everywhere on the plane) and the light vector. For a point close to the light source the light vector and the normal vector are more or less equal resulting in the maximum brightness. The further away, the angle between these vectors increases resulting in decreasing brightness. Just what we expect from a point light source. (The visualization shown here has been altered to improve the visibility of this effect.)

The bottom row shows the lighting mode `'parallel'` with a direction orthogonally to the plane with maximum brightness (left image) and a more and more tilted direction (middle and right images). Since normal vectors and lighting vectors are constant all over the plane, we see constant brightness all over the single images, so a proper parallel lighting.

![](images/plane.png)
