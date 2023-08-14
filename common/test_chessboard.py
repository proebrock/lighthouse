import sys
import os
import pytest
import matplotlib.pyplot as plt



sys.path.append(os.path.abspath('../'))
from common.chessboard import Chessboard
from trafolib.trafo3d import Trafo3d



def test_calibrate_intrinsics():
    board = Chessboard()
    board.plot2d()
    plt.show()

