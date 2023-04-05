import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from common.image_utils import image_load_multiple, image_show_multiple
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel



images = image_load_multiple('image????.png')
print(images.shape)
#image_show_multiple(images, single_window=True)

board = CharucoBoard()
print(board)
print(board.generate_image().shape)
board.plot2d()
#board.plot3d()
#screen = board.generate_screen()
#o3d.visualization.draw_geometries([ \
#    screen.get_cs(100), screen.get_mesh()])

cam = CameraModel()
reprojection_error, trafos = board.calibrate(images, cam, verbose=False)
print(cam)
with np.printoptions(precision=3, suppress=True):
    for trafo in trafos:
        rvec = trafo.get_rotation_rodrigues()
        tvec = trafo.get_translation()
        print(tvec.T.ravel(), rvec.T.ravel())
print(reprojection_error)

with np.printoptions(precision=3, suppress=True):
    for i in range(images.shape[0]):
        trafo = board.estimate_pose(images[i], cam)
        dt, dr = trafos[i].distance(trafo)
        print(dt, np.rad2deg(dr))

image = board.generate_image()
trafo = board.estimate_pose(image, cam, True)
print(trafo)


plt.show()
