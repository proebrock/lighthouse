# Lighthouse

*Lighthouse* is a set of tools to simulate a camera to take pictures of a scene consisting of 3D objects modelled as triangle meshes.

I work as a lecturer at the University of Applied Sciences of the Grisons [FHGR](https://www.fhgr.ch/). I teach Computer Vision classes to Bachelor students. My students use Python and standard computer vision libraries like [OpenCV](https://opencv.org/) to solve computer vision problems. For example a couple of calibrated time-of-flight (ToF) cameras looking at a single moving feature and calculating multiple bundle adjustments in 2D or 3D. To provide example data for those exercises I would need expensive hardware triggered ToF-cameras and lenses. And setting it all up I would still have no idea about the ground truth, e.g. the real position of the feature.

With the *Lighthouse* framework I can set up any kind of static or dynamic scene of 3D objects. Then I can define a couple of cameras 2D or 3D, define their model parameters and place them in the scene. Then I can take pictures with those cameras. This data is basics for the solutions to calculate.

## Features

* Camera model is based on OpenCV model including extrinsics (pose) and intrinsics (chip size, focal length (fx, fy), principal point (cx, cy), radial distortion (k1-k6), tangential distortion (p1, p2) and thin prism distortion (s1-s4))
* Camera provides gray/color image, depth image and colored point cloud; to emulate a simple 2D camera you can omit parts of the data

* Objects: Triangle meshes

## Requirements

## Basics

[trafolib](trafolib)

camsimlib

demo

## Example Applications

