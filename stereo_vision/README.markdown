# Lighthouse: Stereo Vision

Usage of two cameras in a stereo configuration to reconstruct a scene in 3D.

## Scene and image generation

For a test scene we use two cameras in a stereo configuration. Then we create four rectangles and place them at different distances of 500mm, 800mm, 1000mm and 1200mm from the cameras. To be able to do stereo-matching later, these rectangles need to have a texture, so we add that.

![](images/scene.png)

We take pictures with both camera of this scene and we slightly vary the cameras to see how the stereo algorithm deals with distortions or slight translational or rotatory deviations of the cameras.

## Solution

We assume the camera parameters to be known for our experiments and just use the parameters of the generation of the scene saved in the JSON files. In a real world scenario the parameters would have to be estimated by a [calibration](../2d_calibrate_stereo).

### Stereo rectification

Let's take a look at the raw images of the scene from both cameras.

![](images/original.png)

We can see, that the cameras are not perfectly aligned, the right camera is mounted in a slightly lower position. This means, that the rows of the left and right images are not perfectly aligned. In the left image, the red line denotes row 297 which runs along the edge of the largest rectangle. The same row 297 in the right image has different content.

First step in stereo vision is the *stereo rectification*. We undistort the images and project both into new 2D images where a row of the left image corresponds to the same row of the right image. We use `cv2.stereoRectify` to calculate the necessary transformations and provide the function with all camera intrinsics and extrinsics. With those transformations we use `cv2.initUndistortRectifyMap` and `cv2.remap` to get the rectified images.

![](images/rectified.png)

### Stereo block matching

Lets take a look at the row 700 in both images

![](images/rectified_row700.png)

If we plot the brightness of those two rows, the result is the following

![](images/brightness_row700.png)

We see that the left square is translated by about 100 pixels between the left and the right image. The right square appears translated by about 150 pixels. This translation is called *disparity* and

![](images/disparity_row700.png)

```math
\mathrm{distance}=\frac{\mathrm{base_line}\cdot\mathrm{focal_length}}{\mathrm{disparity}}
```


![](images/distance_row700.png)


Stereo block matching is highly dependent on the distances in the image to be expected, the block sizes, the textures, and so on. There are a lot of parameters to optimize stereo block matching. The two scripts [stereo_bm_gui.py](stereo_bm_gui.py) and [stereo_sgbm_gui.py](stereo_sgbm_gui.py) load the rectified images saved by [run.py](run.py) and offer simple sliders to vary stereo block matching parameters and directly observe the resulting disparity image.




Kaehler, A. & Bradski, G. Learning OpenCV 3 O'Reilly UK Ltd., 2017, Chapter 19 "Stereo Imaging"

