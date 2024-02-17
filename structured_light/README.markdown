# Lighthouse: 3D Reconstruction with Structured Light

## The problem

3D reconstruction using structured lighting and one or multiple cameras is a well established method. Let's try if we can realize a 3D reconstruction using the Lighthouse framework.

We use a 3D mesh as the model of the object we want to reconstruct. In the center we place a projector that can project light patterns onto the 3D model (big coordinate system). To the left and right we place two cameras that can take images of the light pattern projected into the object (smaller coordinate systems).

![](images/setup.png)

At this point we assume that the intrinsics and extrinsics of the camera and the projector are both known. These have to be estimated by calibration. See the chapter about projector calibration.

## Correspondence matching

### Patterns

One key problem to solve is *correspondence matching*. If we can match camera pixels to projector pixels, we have bundles of projector and camera rays meeting at a point at the surface of the object. We can determine the 3D location of this point using *bundle adjustment*.

For correspondence matching, we display a sequence of images with the projector and take pictures with each camera. This way we are able to encode each projector pixel with a unique sequence of brightness values over time. Usually row and column of each pixel are identified separately. The sequence could be a black and white pattern that identifies each pixel using a *binary encoding*.

![](images/pattern_binary.gif)

But in this example we use a *phase shift encoding*. We use images with a sinusoidal change of brightness. On the left we see the sequence of patterns over time and on the right how the images of the right camera look like when projecting the phase shift pattern onto the object.

![](images/pattern_phase.gif)
![](images/images.gif)

The left image shows the reconstructed rows for each pixel in color encoding. The cursor is located at a single pixel. The right plot shows the brightness of this single pixel over time (11 images). The changing brightness follows a sinusoidal function, the phase shift of that sinus curve encodes the row.

![](images/sine_fit.png)

Finally we take two additional images with the projector turned off and one with the projector lighting the object with bright white light.

![](images/image_black.png)
![](images/image_white.png)

The images can be used for segmenting where in the image the object is or for color sampling to colorize the finally reconstructed point cloud.

### Results

The correspondence matching fits a sinus curve through each pixel in the time domain and estimates the phase. The phase can be translated into a position (row or column) with subpixel accuracy. So for each camera pixel we have a matching sub-pixel on the projector chip.

This image shows a section of the projector chip with black lines marking the pixels of the projector. The places different camera pixels match to are marked with crosses in different colors for different cameras.

![](images/projector_low.png)

In order to do bundle adjustment, we can use a single cross: we use the camera pixel that maps to that cross and the sub-pixel projector chip location denoted by the cross. That gives two rays and is the minimal amount of information to do bundle adjustment. It does not allow for an error estimate. If two crosses of different cameras would be close together on the projector chip, this would very much improve the situation: bundle adjustment with 3 rays including an error estimate. But the density of mappings does not really allow us to find two crosses of different cameras next to each other.

The low density of crosses depends on various factors such as the resolutions of camera and projector, their distances to the object and of course local properties of the object. Something we can easily do is to increase the camera resolution in our simulation.

![](images/projector_high.png)

Now we have a much higher density of crosses and can improve our bundle adjustment. But now we even have multiple crosses of the same camera next to each other... What does that mean? Where do this matches come from?

Another way to visualize the matching is to plot a section of a single camera chip (left) and the matching section of the projector chips (right) next to each other. In the right plot we summarize all crosses that map to a pixel with a counter (color encoded). If we put the cursor to a projector pixel, the left plot shows in red all camera pixels mapping to that projector pixel (in yellow camera pixels that have a mapping). In the low resolution scenario this is easy: low density of points on the projector chip, each projector pixel is mapped by a single camera pixel.

![](images/matching_low.png)

In the high resolution scenario this is different. Most projector pixels are mapped by multiple camera pixels. Those camera pixels are usually in a limited area but are not necessarily round (see red points). This effect has to be kept in mind when selecting matches for bundle adjustment.

![](images/matching_high.png)

Based on these insight, we implement the 3d reconstruction as follows: we cluster points based on the pixels on the projector chip. On the projector chip we calculate the center of gravity of the matches, this gives the projector ray. For each camera pixel matching to this projector chip pixel, we calculate the center of gravity of those pixels. This gives one or multiple camera rays.

## Reconstruction

With the matching 2d points (rays) on projector and cameras we can run a bundle adjustment to make a reconstruction of the object (low resolution version shown here):

![](images/pointcloud_low.png)

Due to the fact that this is a simulation, we can compare the reconstructed point cloud with the original mesh of the object to get an estimate on the quality of the reconstruction (e.g. distances between points and the mesh).

