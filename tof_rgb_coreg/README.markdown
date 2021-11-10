# Lighthouse: ToF and RGB data co-registration

Time-of-flight cameras (ToF) provide depth and intensity information, but no color information. Combining a ToF camera with a standard color camera in a stereo configuration and the co-registration of the data would provide both. This project simulates such a camera combination and offers an algorithm for co-registration.

## Problem description

The simulated scene consists of the 3D model (mesh with vertex colors) of an object (a fox head) and two cameras, a ToF camera and a RGB camera in a stereo configuration.

![](images/scene.png)

The extrinsic camera parameters (the 3D transformation between the cameras) are considered to be known as well as the intrinsic camera parameters of the RGB camera. For a real world application, these parameters could be determined by calibration.

### ToF camera

The ToF camera image consists of a point cloud. The reflectivity of the object for the infrared light gives an additional intensity information per point (displayed as gray value).

![](images/pcl_gray_scaled.png)

### RGB camera

The RGB camera provides an RGB raster image of the scene from the viewpoint of the RGB camera.

![](images/rgb_image_scaled.png)

## Solution

### Projection on RGB chip

Key idea is to use the 3D points provided by the ToF camera. Using the extrinsics of the cameras we can transform these points into the coordinate system of the RGB camera. Then we can project these 3D points onto the camera chip of the RGB camera.

These 2D points do not necessarily end up exactly in the center of the pixels of the camera chip of the RGB camera. For determining the final color to a 2D point and the corresponding 3D point, some two dimensional interpolation is required.

![](images/rgb_image_points_scaled_zoom.png)

Since both cameras have different focal lengths an different view points, it is not guaranteed that each point of the ToF camera ends up on the chip of the RGB camera.

![](images/rgb_image_points_scaled.png)

For these points we cannot determine the color of the 3D point, since the RGB camera does not see them.

### Consistency check: View angle

There is some possible mismatching of points to colors using the approach described above. An example: The ToF camera is located on the left and may pick up a point on the left side of the snout of the fox (view of the observer) which may be out of view of the RGB camera. Transforming this point to the camera chip would yield a color, but one from a point on the right side of the snout visible for the RGB camera.

This problem can be solved by making sure that we are actually looking with the RGB camera on the "outer" side of the surface sampled by the points of the ToF camera. The solution to determine the outer side can be solved by calculating the normal vectors for each point of the point cloud that sit orthogonally on the surface of the object and oriented towards the RGB camera.

![](images/pcl_gray_normals_scaled.png)

The problem can now be expressed differently: To make sure we look on the outside of the surface of the object, for each point we have to make sure that the angle between its normal vector and the directional vector to the optical center of the RGB camera is lower or equal than 90 degrees. Otherwise the RGB camera does look onto the underside of the surface.

### Final result

This image shows the final result of the co-registration. All points in the point cloud have proper colors apart from red points that are out of view of the RGB camera and green points where the RGB camera looks at the underside of the surface of the object.

![](images/pcl_result.png)

### Quality

Since we work with simulated data, we can do something special. The simulated ToF camera does indeed capture the colors of the points, even though this data is removed to provide realistic basis for the experiment. Now we can use these colors as ground truth and compare them with the estimated point colors from the RGB camera!

For comparing the colors we just use a simple metric by calculating the vector distance between two RGB vectors. Shown as a heat map the result looks like this:

![](images/pcl_quality.png)

First we see some light variations of blue on the face of the fox. This is caused by differences in lighting. The current `CameraModel` assumes a point light source at the origin of the camera coordinate system. Since this means that the light source was different for both cameras and both images taken, we get some slight error in color. Once the `CameraModel` supports setting the lighting, we can have global lighting and fix this problem (**TODO**).

More pronounced is an error on the left side of the snout. In the reconstructed point cloud this is visible as a black line.

![](images/snout_error.png)

Reason for this error is the view direction of the RGB camera which looks right over edge the snout (view from RGB camera).

![](images/snout_reason.png)

So some points of the ToF point cloud are projected on pixels on the snout that are darker, some are projected on pixels of the cheek that are brighter resulting in this black line. Effects like this can be expected to be more drastic in case of real world cameras and the presence of noise.
