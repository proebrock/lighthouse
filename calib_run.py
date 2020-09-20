import cv2
import cv2.aruco as aruco
import glob


aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
board = aruco.CharucoBoard_create(4, 5, 60.0, 30.0, aruco_dict)
parameters = aruco.DetectorParameters_create()


allCorners = []
allIds = []

imageSize = None

for fname in sorted(glob.glob('*.png')):
    print('Calibration using ' + fname + ' ...')

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageSize = gray.shape

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict,
        parameters=parameters)

    aruco.drawDetectedMarkers(img, corners, ids)

    corners, ids, rejected, recovered_ids = aruco.refineDetectedMarkers( \
        gray, board, corners, ids, rejected)

    charuco_retval, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco( \
        corners, ids, gray, board)

    if charuco_corners is not None and charuco_corners.shape[0] >= 4:
            allCorners.append(charuco_corners)
            allIds.append(charuco_ids)
    else:
        print('    Image rejected.')

    aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)

#    img = cv2.resize(img, (0,0), fx=1.0, fy=1.0)
#    cv2.imshow('image', img)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()



print('Calculating calibration ...')
flags = 0
flags |= cv2.CALIB_FIX_K1
flags |= cv2.CALIB_FIX_K2
flags |= cv2.CALIB_FIX_K3
flags |= cv2.CALIB_FIX_K4
flags |= cv2.CALIB_FIX_K5
flags |= cv2.CALIB_FIX_K6
flags |= cv2.CALIB_ZERO_TANGENT_DIST
#flags |= cv2.CALIB_FIX_ASPECT_RATIO
#flags |= cv2.CALIB_RATIONAL_MODEL
reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs = \
    cv2.aruco.calibrateCameraCharuco(allCorners, allIds, \
    board, imageSize, None, None, flags=flags)
print('Reprojection error is {0:.2f}'.format(reprojection_error))
print('Camera matrix is ')
print(camera_matrix)
print('Distortion coefficients')
print(dist_coeffs)

