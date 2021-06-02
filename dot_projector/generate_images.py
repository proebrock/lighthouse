import matplotlib.pyplot as plt
plt.close('all')
import os
import sys

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from dot_projector import DotProjector



if __name__ == '__main__':
    proj = DotProjector()
    proj.plot_chip()