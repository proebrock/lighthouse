import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.mesh_object import MeshObject
from camsimlib.charuco_board import CharucoBoard
from camsimlib.scene_visualizer import SceneVisualizer



def show_images(depth_image, color_image, cbar_enabled=False):
    # Color of invalid pixels
    nan_color = (0, 0, 1.0)
    fig = plt.figure()
    # Depth image
    ax = fig.add_subplot(121)
    cmap = plt.cm.viridis_r
    cmap.set_bad(color=nan_color, alpha=1.0)
    im = ax.imshow(depth_image, cmap=cmap)
    if cbar_enabled:
        fig.colorbar(im, ax=ax)
    ax.set_axis_off()
    ax.set_title('Depth')
    ax.set_aspect('equal')
    # Color image
    idx = np.where(np.isnan(color_image))
    color_image[idx[0], idx[1], :] = nan_color
    ax = fig.add_subplot(122)
    ax.imshow(color_image)
    ax.set_axis_off()
    ax.set_title('Color')
    ax.set_aspect('equal')
    # Show
    plt.show()


if __name__ == '__main__':
    if True:
        mesh = MeshObject()
        #mesh.load('../data/pyramid.ply')
        mesh.load('../data/sphere.ply')
        mesh.scale(100.0)
        #mesh.load('../data/knot.ply')
        #mesh.load('../data/cube.ply')
        #mesh.generate_from_image_file('../data/tux.png', 2.0)
        #mesh.demean()
        mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
        #mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
    else:
        mesh = CharucoBoard((3,4), 40.0)
        mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
        #mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))
    #mesh.show(True, False, False)

    cam = CameraModel((40, 30), 40, camera_pose=Trafo3d(t=(0,0,-500)))
    cam.scale_resolution(1)

    tic = time.process_time()
    depth_image, color_image, P = cam.snap(mesh)
    toc = time.process_time()
    print(f'Snapping image took {(toc - tic):.1f}s')
    show_images(depth_image, color_image)

    vis = SceneVisualizer()
    vis.add_mesh(mesh)
    vis.add_cam_cs(cam, size=100.0)
    vis.add_points(P)
    vis.add_cam_rays(cam, P)
    vis.add_cam_frustum(cam, size=600.0)
    vis.show()
