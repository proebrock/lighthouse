import numpy as np
from cameraModel import CameraModel
from meshObject import MeshObject
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


mesh = MeshObject()
mesh.load('data/knot.ply', demean=True)
#mesh.show()

width = 400
height = 400
cam = CameraModel((width, height), (800, 800))
tic = time.process_time()
dImg, cImg, P = cam.snap(mesh)
toc = time.process_time()
print(f'Snapping image took {(toc - tic):.1f}s')

fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(np.transpose(cImg, axes=(1,0,2)))
ax.set_axis_off()
ax.set_title('Color')
ax.set_aspect('equal')
ax = fig.add_subplot(122)
ax.imshow(dImg.T)
ax.set_axis_off()
ax.set_title('Depth')
ax.set_aspect('equal')
plt.show()

