import cv2
import cv2.aruco as aruco
import numpy as np

aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
squaresX = 2
squaresY = 3
squareLength = 10
markerLength = 5
board = aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, aruco_dict)

sidePixels = 120
img = board.draw((squaresX * sidePixels, squaresY * sidePixels))

cv2.imwrite('charuco_pattern.png', img)

cv2.imshow('ArUco', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

