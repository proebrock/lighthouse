from common.image_utils import image_load_multiple, image_show_multiple
from common.aruco_utils import CharucoBoard


images = image_load_multiple('image????.png', black_white=True)
board = CharucoBoard((5, 7), 80, 20, 10)
print(board)
board.plot3d()