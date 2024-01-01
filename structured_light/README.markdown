# Lighthouse: 3D Reconstruction with Structured Light

## The problem

3D reconstruction using structured lighting and one or multiple cameras is a well established method. Let's try if we can realize a 3D reconstruction using the Lighthouse framework.

We use a 3D mesh as the model of the object we want to reconstruct. In the center we place a projector that can project light patterns onto the 3D model (big coordinate system). To the left and right we place two cameras that can take images of the light pattern projected into the object (smaller coordinate systems).

![](images/setup.png)

At this point we assume that the intrinsics and extrinsics of the camera and the projector are both known.

## Correspondence matching

### Patterns

One key problem to solve is *correspondence matching*. If we can match camera pixels to projector pixels, we have bundles of projector and camera rays meeting at a point at the surface of the object. We can determine the 3D location of this point using *bundle adjustment*.

For correspondence matching, we display a sequence of images with the projector and take pictures with each camera. This way we are able to encode each projector pixel with a unique sequence of brightness values over time. Usually row and column of each pixel are identified separately. The sequence could be a black and white pattern that identifies each pixel using a *binary encoding*.

![](images/pattern_binary.gif)

But in this example we use a *phase shift encoding*. We use images with a sinusoidal change of brightness. On the left we see the sequence of patterns over time and on the right how the images of the right camera look like when projecting the phase shift pattern onto the object.

![](images/pattern_phase.gif)
![](images/images.gif)

![](images/sine_fit.png)

![](images/image_black.png)
![](images/image_white.png)

### Results

#### Low resolution camera coverage

![](images/matching_low.png)
![](images/projector_low.png)

#### High resolution camera coverage

![](images/matching_high.png)
![](images/projector_high.png)

## Reconstruction

![](images/pointcloud_low.png)

