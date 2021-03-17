# Hand eye calibration

In industrial robotics, we often use sensors to adapt the robot program to the actual position of the work object. In an example application we have barrels on a conveyor belt and the robot has to put a nozzle through the bung hole of the barrel to fill up the barrel with liquid. If the barrel stops on the conveyor belt, its location has an accuracy of +-100mm and the barrel is be rotated by an arbitrary angle. So the position of the bung hole varies greatly. An obvious solution would be to use a camera system to detect the position of the bung hole and transfer it to the robot in order to fit the nozzle through it. Unfortunately, if we detect a feature in the camera coordinate system, we do not know how to translate this information into robot coordinate system, the robot moves in. We need to find the translation from the camera coordinate system to the robot coordinate system. This problem of finding this transformation is called [hand-eye-calibration](https://en.wikipedia.org/wiki/Hand_eye_calibration_problem).

Theoretically, there are two basic configurations, one where the camera is mounted in the robot hand and one where the camera is static. The hand-eye-calibration for both is very similar. We take a closer look at the configuration with the robot mounted camera.

![](images/cam_mobile.png)


![](images/setup.jpg)
