import matplotlib.pyplot as plt
plt.close('all')
import os
import sys
import numpy as np
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from dot_projector import DotProjector
from camsimlib.camera_model import CameraModel



if __name__ == '__main__':
    np.random.seed(42) # Random but reproducible

    proj = DotProjector()
#    proj.plot_chip()

    cam = CameraModel(chip_size=(80, 60), focal_length=(80, 88))
    cam.set_camera_pose(Trafo3d(t=(200, 0, 0), rpy=np.deg2rad((0, -25, 0))))

    plane = Trafo3d(t=(0, 0, 400), rpy=np.deg2rad((10, 180, -50)))
    P = proj.intersect_plane(plane)
    p = cam.scene_to_chip(P)[:, 0:2]
    # Reduce p to just the valid points
    indices = np.round(p).astype(int)
    x_valid = np.logical_and(indices[:, 0] >= 0, indices[:, 0] < cam.chip_size[0])
    y_valid = np.logical_and(indices[:, 1] >= 0, indices[:, 1] < cam.chip_size[1])
    valid = np.logical_and(x_valid, y_valid)
    p = p[valid,:]
    print(f'Number of invalid points (out of chip): {np.sum(~valid)}')

#    # Plot points on camera chip
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    ax.plot(p[:,0], p[:,1], 'ob')
#    ax.grid()
#    plt.show()

#    # Visualize setup
    proj_cs = proj.get_cs(size=50)
    proj_frustum = proj.get_frustum(size=200)
    cam_cs = cam.get_cs(size=100)
    cam_frustum = cam.get_frustum(size=200)
    plane_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
    plane_cs.transform(plane.get_homogeneous_matrix())
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(P)
    pcl.paint_uniform_color((0.8, 0.2, 0.2))
    np.asarray(pcl.colors)[valid,:] = ((0.2, 0.8, 0.2))

#    o3d.visualization.gui.Application.instance.initialize()
#    w = o3d.visualization.O3DVisualizer('Dot projector setup', 800, 600)
#    w.add_geometry('Projector CS', proj_cs)
#    w.reset_camera_to_default()
#    o3d.visualization.gui.Application.instance.add_window(w)
#    o3d.visualization.gui.Application.instance.run()

    o3d.visualization.draw_geometries([proj_cs, proj_frustum,
                                       cam_cs, cam_frustum,
                                       plane_cs, pcl])

