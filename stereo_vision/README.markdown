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

First step in stereo vision is the **stereo rectification**. We undistort the images and project both into new co-planar 2D images where a row of the left image corresponds to the same row of the right image. We use `cv2.stereoRectify` to calculate the necessary transformations and provide the function with all camera intrinsics and extrinsics. With those transformations we use `cv2.initUndistortRectifyMap` and `cv2.remap` to get the rectified images.

![](images/rectified.png)

### Disparity and distance

Lets take a look at the row 700 in the left and right image

![](images/rectified_row700.png)

If we plot the brightness of those two rows, the result is the following

![](images/brightness_row700.png)

We see that the left square is translated by about 100 pixels between the left and the right image. The right square appears translated by about 150 pixels. This translation is called **disparity**.

The disparity is dependent on the distance of the object from the cameras

![](images/disparity_drawing.png)

We have to cameras, both looking in the same direction and with a baseline distance of $`b`$. The cameras have an identical focal length of $`f`$. The object (black circle) is in a distance of $`z`$. The disparity in this configuration - object between optical axes of both cameras - is $`d=d_1+d_2`$. The variable $`x`$ is temporary.

On the left side we have

```math
\frac{x}{z}=\frac{d_1}{f}
```

And on the right side

```math
\frac{b-x}{z}=\frac{d_2}{f}
```

Solving for $`x`$ and putting things together:

```math
\frac{d_1\cdot z}{f}=b-\frac{d_2\cdot z}{f}\quad\Leftrightarrow\quad d=d_1+d_2=\frac{b\cdot f}{z}
```

The disparity is reciprocally proportional to the distance of the object: the further the object away, the smaller the disparity. And if we know the disparity and the other parameters like focal length and baseline distance, we can calculate the distance of the objects.

For our example here we have $`f=1500\mathrm{pix}`$, $`b=80\mathrm{mm}`$. The distance for a disparity of $`d=100\mathrm{pix}`$ as measured above we get

```math
z=\frac{b\cdot f}{d}=\frac{80\mathrm{mm}\cdot 1500\mathrm{pix}}{100\mathrm{pix}}=1200\mathrm{mm}
```

For a disparity of $`d=150\mathrm{pix}`$ we get a distance of $`z=800\mathrm{pix}`$.

### Stereo block matching






![](images/disparity_row700.png)


From the disparity we can calculate the distance to the object


![](images/distance_row700.png)



Stereo block matching is highly dependent on the distances in the image to be expected, the block sizes, the textures, and so on. There are a lot of parameters to optimize stereo block matching. The two scripts [stereo_bm_gui.py](stereo_bm_gui.py) and [stereo_sgbm_gui.py](stereo_sgbm_gui.py) load the rectified images saved by [run.py](run.py) and offer simple sliders to vary stereo block matching parameters and directly observe the resulting disparity image.




Kaehler, A. & Bradski, G. Learning OpenCV 3 O'Reilly UK Ltd., 2017, Chapter 19 "Stereo Imaging"

