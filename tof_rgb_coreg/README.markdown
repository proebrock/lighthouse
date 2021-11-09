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

![](images/rgb_image_points_scaled.png)

Since both cameras have different focal lengths an different view points, it is not guaranteed that each point of the ToF camera ends up on the chip of the RGB camera.

![](images/rgb_image_points_scaled_zoom.png)

For these points we cannot determine the color of the 3D point, since the RGB camera does not see them.

### Consistency check: View angle

![](images/pcl_gray_normals_scaled.png)

### Final result

![](images/pcl_result.png)
