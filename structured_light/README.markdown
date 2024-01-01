# Lighthouse: 3D Reconstruction with Structured Light

## The problem

3D reconstruction using structured lighting and one or multiple cameras is an established method. Let's try if we can realize a 3D reconstruction using the Lighthouse framework.

We use a 3D mesh as the model of the object we want to reconstruct. In the center we place a projector that can project light patterns onto the 3D model (big coordinate system). To the left and right we place two cameras that can take images of the light pattern projected into the object (smaller coordinate systems).

![](images/setup.png)

At this point we assume that the intrinsics and extrinsics of the camera and the projector are both known.

## Correspondence matching

### Patterns

![](images/image_black.png)
![](images/image_white.png)

![](images/pattern_phase.gif)
![](images/images.gif)

![](images/sine_fit.png)

![](images/pattern_binary.gif)

### Results

#### High resolution camera coverage

![](images/matching_high.png)
![](images/projector_high.png)

#### Low resolution camera coverage

![](images/matching_low.png)
![](images/projector_low.png)

## Reconstruction

![](images/pointcloud_low.png)

