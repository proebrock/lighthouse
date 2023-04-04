import open3d as o3d
import matplotlib.pyplot as plt
plt.close('all')

from common.image_utils import image_load_multiple, image_show_multiple
from common.aruco_utils import CharucoBoard
from camsimlib.camera_model import CameraModel



images = image_load_multiple('image????.png')
print(images.shape)
image_show_multiple(images, single_window=True)

board = CharucoBoard((5, 7), 80, 20, 10)
print(board)
#board.plot2d()
#board.plot3d()
#screen = board.generate_screen()
#o3d.visualization.draw_geometries([ \
#    screen.get_cs(100), screen.get_mesh()])

cam = CameraModel()
board.calibrate(images, cam, verbose=True)
print(cam)
plt.show()
