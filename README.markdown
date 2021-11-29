# Lighthouse

*Lighthouse* is a set of tools to **simulate a camera** to take pictures of a scene consisting of 3D objects modeled as triangle meshes.

I work as a lecturer and teach Computer Vision classes to Bachelor students. My students use Python and standard Computer Vision libraries like [OpenCV](https://opencv.org/) to solve Computer Vision problems. For example a couple of calibrated time-of-flight (ToF) cameras looking at a single moving feature and calculating multiple bundle adjustments in 2D or 3D. To provide example data for those exercises I would need expensive hardware triggered ToF-cameras and lenses. And setting it all up I would still have no idea about the ground truth, e.g. the real position of the feature.

With the *Lighthouse* framework I can set up any kind of static or dynamic scene of 3D objects. Then I can define a couple of cameras 2D or 3D, define their model parameters and place them in the scene. Then I can take pictures with those cameras. This data is basics for the solutions to calculate.

Finally, since the whole framework is implemented in Python and makes some interesting learning material about the internal workings of a camera model or a ray tracer.



## Features

* Camera model is based on OpenCV model including extrinsics (pose) and intrinsics (chip size (width, height), focal length (fx, fy), principal point (cx, cy), radial distortion (k1-k6), tangential distortion (p1, p2) and thin prism distortion (s1-s4))
* Camera provides gray/color image, depth image and colored point cloud; to emulate a simple 2D camera you can omit parts of the data
* Flat or Gouraud shading
* Objects: Triangle meshes, supports vertex colors; textures not supported at the moment
* Snapping an image calculates a ray from each pixel of the camera and intersects it with all triangles of the scene; raytracer is implemented in Python and uses multiprocessing; still, ray tracing is slow
* Lots of Computer Vision example applications for educational purposes
* Permissive License (MIT)



## Installation

### Requirements

* [Python](https://www.python.org/) 3.6+
* `NumPy`, `Matplotlib`, `SciPy`
* [OpenCV](https://opencv.org/) 4.5+ with `contrib` packages
* [Open3D](http://www.open3d.org/) 0.13.0+ for 3D model handling and visualizations
* `PyTest` for the testsuite

Installation via `pip`:

```
pip install open3d opencv-contrib-python scipy matplotlib pytest h5py
```

### Usage

Just checkout the stable `main` branch of the repository and you are good to go!



## Basics

To handle 2D and 3D transformations and their different flavors (matrices, Euler angles, quaternions, Rodrigues vectors, etc.), we use the [trafolib](trafolib) library with its classes `Trafo2d` and `Trafo3d`. The latter uses [pyquarternion](https://github.com/KieranWynn/pyquaternion) for its internal representation of rotations. All classes are well-tested and well-documented. The library has few external dependencies and can easily be used for any project requiring computations on transformations.

The workhorse of this framework is the [camsimlib](camsimlib) library with its main class `CameraModel`. It provides the camera model as well as functionality to snap images from 3D scenes.

In the [demo](demo) directory you can find a simple minimal example showing you the capabilities of the *Lighthouse* framework.



## Example Applications

|Application                                           |Description                                               |
|------------------------------------------------------|----------------------------------------------------------|
|[demo](demo)                                          |Demo projects to show basic functionality of the framework|
|[2d_calibrate_single](2d_calibrate_single)            |Calibrate a single cam                                    |
|[2d_calibrate_multiple](2d_calibrate_multiple)        |Calibrate multiple cams (intr.+extr.)                     |
|[bundle_adjust_simple](bundle_adjust_simple)          |Four cams watching single feature, run bundle adjust      |
|[multi-marker](multi-marker)                          |Object with multiple markers on it, detect its pose       |
|[bundle_adjust_trajectory](bundle_adjust_trajectory)  |Feature moves, reconstruct trajectory                     |
|[2d_ball_locate](2d_ball_locate)                      |Detect the 3D position of a ball with a single cam        |
|[hand_eye_calib_2d](hand_eye_calib_2d)                |Robot hand-eye calibration                                |
|[tof_rgb_coreg](tof_rgb_coreg)                        |ToF camera and RGB camera integration/co-registration     |



## Work in progress and/or not documented

|Application                                           |Description                                               |
|------------------------------------------------------|----------------------------------------------------------|
|[bundle_adjust_large_scale](bundle_adjust_large_scale)|Multiple objects, reconstruct movement of single cam      |
|[dot_projector](dot_projector)                        |--Under construction--                                    |



## About me and the project

My name is Philipp Roebrock and I work as a lecturer at the Institute for Photonics and ICT at the University of Applied Sciences of the Grisons [FHGR](https://www.fhgr.ch/) in Chur in Switzerland. I will work on this project during my Computer Vision classes in spring semester. The amount of work I can invest into this framework changes. But I have a long list of TODOs I want to implement.

If you have any questions or suggestions please please drop and issue or a personal message to `philipp DOT roebrock AT fhgr DOT ch`.
