# Lighthouse: Stereo Vision

Usage of two cameras in a stereo configuration to reconstruct a scene in 3D.

## Scene and image generation

For a test scene we use two cameras in a stereo configuration. Then we create four rectangles and place them at different distances from the cameras. To be able to do stereo-matching later, these rectangles need to have a texture, so we add that.

![](images/scene.png)

We take pictures with both camera of this scene and we slightly vary the cameras to see how the stereo algorithm deals with distortions or slight translational or rotatory deviations of the cameras.

## Solution

We assume the camera parameters to be perfectly known for our experiments and just use the parameters of the generation of the scene. In a real world scenario the parameters have to be estimated by a [calibration](../2d_calibrate_stereo).







Kaehler, A. & Bradski, G. Learning OpenCV 3 O'Reilly UK Ltd., 2017, Chapter 19 "Stereo Imaging"

