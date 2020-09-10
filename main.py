import numpy as np
from trafolib.Trafo3d import Trafo3d
from CameraModel import CameraModel
from MeshObject import MeshObject
import matplotlib.pyplot as plt
import time
import cv2


mesh = MeshObject()

if True:
	#mesh.generateFromImageFile('data/tux.png', 2.0)
	mesh.generateChArUco((3,2), 50.0)
	#mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
else:
	mesh.load('data/pyramid.ply')
	mesh.demean()
	mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))

mesh.show(True, False, False)

if False:
	cam = CameraModel((400, 400), 800, T=Trafo3d(t=(0,0,-500)))
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
	im = ax.imshow(dImg.T)
	#fig.colorbar(im, ax=ax)
	ax.set_axis_off()
	ax.set_title('Depth')
	ax.set_aspect('equal')
	plt.show()

