# Hand eye calibration

In industrial robotics, we often use sensors to adapt the robot program to the actual position of the work object. In an example application we have barrels on a conveyor belt and the robot has to put a nozzle through the bung hole of the barrel to fill up the barrel with liquid. If the barrel stops on the conveyor belt, its location has an accuracy of +-100mm and the barrel is be rotated by an arbitrary angle. So the position of the bung hole varies greatly. An obvious solution would be to use a camera system to detect the position of the bung hole and transfer it to the robot in order to fit the nozzle through it. Unfortunately, if we detect a feature in the camera coordinate system, we do not know how to translate this information into robot coordinate system, the robot moves in. We need to find the translation from the camera coordinate system to the robot coordinate system. This problem of finding this transformation is called [hand-eye-calibration](https://en.wikipedia.org/wiki/Hand_eye_calibration_problem).

Theoretically, there are two basic configurations, one where the camera is mounted in the robot hand and one where the camera is static. The hand-eye-calibration for both is very similar. We take a closer look at the configuration with the robot mounted camera.

For the calibration, we put a calibration board at a fixed place in front of the robot and make the robot place the camera at `n` different poses and take images of the calibration board. Let's have a look at the transformations in this scenario:

![](images/cam_mobile.png)

The base coordinate system of the robot is our world coordinate system. The board is located at a fixed position `base_T_board`. We consider this position as unknown. The `n` transformations `base_T_flange,i` are determined by the robot poses in the robot program and are therefore known. The transformation `flange_T_cam` is fixed and is the transformation we want to determine in the hand eye calibration. This transformation that enables us to transform features in the camera coordinate system into the robot base coordinate system (when the robot pose is known). The transformation could be seen as the *tool transformation* of the camera in the robots hand. Finally the transformations `cam_T_board,i` are the `n` known transformations resulting as the extrinsic camera parameters of the camera calibration we can calculate for every of the `n` images.

Here is an image of an example configuration with a small ABB robot and a Raspberry Pi High Resolution Camera. The coordinate systems are shown in the RGB-XYZ notation. The camera is upside down and looks to the left and shows the image on the paper in a proper way (TL=top-left,BR=bottom-right, ...).

![](images/setup.jpg)

Even though the example images are generated using the camsimlib, it is based on the real-world example shown above.

Okay. How can we determine the tool transformation of the camera `flange_T_cam`?

## Using numerical optimization

Let's sketch a solution using numerical optimization!

The two unknown transformations are `flange_T_cam` (wanted) and `base_T_board` (by-product). If we express those as translation vector (3 components) and Rodrigues vector (3 components) each, we have 12 unknowns in our **decision variable**.

As **initial values** we can use rough measures from our real world example. But initializing all 12 values with zeros should do fine.

The **objective function** takes the decision variables as parameter and additionally the two lists `base_T_flange,i` and `cam_T_board,i` containing `n` transformations each. It calculates `n` times the transformation `base_T_board` using `base_T_flange,i`, `flange_T_cam` and `cam_T_board,i` and compares it with the real transformation `base_T_board`. For the comparison we can use `Trafo3d.distance`, that calculates the distance of two transformations. For the translational part it returns with `dt`the distance of both coordinate systems, for the rotational part the overall rotation angle `dr`. We can just sum up all `n` instances of `dt` and `dr` separately. For the return value we need to return one value from the sums of `dt` and `dr`. We can convert the angle in degrees and add it to the translational distance. This way the optimization considers a deviation of 1mm same as a deviation of 1deg.

For the **optimizer** use `minimize` from `scipy.optimize`. Use the Nelder-Mead algorithm and enable the flag `adaptive`.

The expected result for `flange_T_cam` is ([ 20, -20, 6], [110., -87., -20.]) (translation, rotation in RPY angles in degrees). Is it plausible? Check the photo of the setup!

## Using OpenCV

The [OpenCV](https://docs.opencv.org/master/) library offers a function for calculating a hand-eye-calibration which is called `cv::calibrateHandEye`. Check out the documentation. Use the `cv::CALIB_HAND_EYE_PARK` algorithm to calculate a solution.

OpenCV uses a different terminology: "flange" is called "gripper" and "board" is called "target".

The result for `flange_T_cam` should be similar: ([ 20, -20, 6], [110., -87., -20.]) (translation, rotation in RPY angles in degrees).

