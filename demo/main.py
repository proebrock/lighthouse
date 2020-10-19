import numpy as np
import matplotlib.pyplot as plt
plt.close('all')
import os
import sys
import time
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, mesh_generate_plane, \
    mesh_generate_image_file, mesh_generate_charuco_board, mesh_generate_rays



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
    cam = CameraModel((40, 30), 40, camera_pose=Trafo3d(t=(0,0,-500)))
    cam.scale_resolution(2)

    plane = mesh_generate_plane((200, 200), color=(1,1,0))
#    plane = mesh_generate_image_file('../data/tux.png', pixel_size=3)
#    plane = mesh_generate_charuco_board((6, 5), 30.0)
    plane.translate(-plane.get_center())
    mesh_transform(plane, Trafo3d(rpy=np.deg2rad([155,25,0])))

    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.scale(0.3, center=sphere.get_center())
    sphere.paint_uniform_color((1,0,0))
    sphere.translate((0,0,-25))

    cs = cam.get_cs(size=100.0)
#    cs = cam.get_frustum(size=600.0)

    o3d.visualization.draw_geometries([cs, plane, sphere])

    mesh = plane + sphere
    tic = time.process_time()
    depth_image, color_image, pcl = cam.snap(mesh)
    toc = time.process_time()
    print(f'Snapping image took {(toc - tic):.1f}s')

    show_images(depth_image, color_image)
    o3d.io.write_point_cloud('out.ply', pcl)

#    rays = mesh_generate_rays(cam.get_camera_pose().get_translation(), pcl, (0,0,0))
#    o3d.visualization.draw_geometries([cs, plane, sphere, rays])
