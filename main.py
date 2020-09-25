import numpy as np
from trafolib.trafo3d import Trafo3d
from camera_model import CameraModel
from mesh_object import MeshObject
from charuco_board import CharucoBoard
import matplotlib.pyplot as plt
plt.close('all')
import time



def show_images(dImg, cImg, cbar_enabled=False):
    # Color of invalid pixels
    nan_color = (0, 0, 1.0)
    fig = plt.figure()
    # Depth image
    ax = fig.add_subplot(121)
    cmap = plt.cm.viridis_r
    cmap.set_bad(color=nan_color, alpha=1.0)
    im = ax.imshow(dImg, cmap=cmap)
    if cbar_enabled:
        fig.colorbar(im, ax=ax)
    ax.set_axis_off()
    ax.set_title('Depth')
    ax.set_aspect('equal')
    # Color image
    idx = np.where(np.isnan(cImg))
    cImg[idx[0], idx[1], :] = nan_color
    ax = fig.add_subplot(122)
    ax.imshow(cImg)
    ax.set_axis_off()
    ax.set_title('Color')
    ax.set_aspect('equal')
    # Show
    plt.show()



if False:
    mesh = MeshObject()
    mesh.load('data/pyramid.ply')
    #mesh.load('data/knot.ply')
    #mesh.load('data/cube.ply')
    mesh.generate_from_image_file('data/tux.png', 2.0)
    #mesh.demean()
    #mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
    #mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
else:
    mesh = CharucoBoard((3,4), 40.0)
    mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
    #mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
#mesh.show(True, False, False)

if True:
    cam = CameraModel((120, 90), 150, trafo=Trafo3d(t=(0,0,-500)))
    print(cam)
    print(mesh)
    tic = time.process_time()
    dImg, cImg, P = cam.snap(mesh)
    toc = time.process_time()
    print(f'Snapping image took {(toc - tic):.1f}s')
    show_images(dImg, cImg)

