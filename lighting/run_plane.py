import matplotlib.pyplot as plt
from matplotlib import colors, ticker, cm
from matplotlib.colors import LogNorm
plt.close('all')
import numpy as np
import os
import sys
import time
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_transform, show_images, save_shot, mesh_generate_image


if __name__ == '__main__':
    # Create camera and set its model parameters
    cam = CameraModel(chip_size=(40, 30),
                      focal_length=(50, 50))
    cam.scale_resolution(4)

    img = 255 * np.ones((100, 100, 3))
    mesh = mesh_generate_image(img, pixel_size=5.0)
    mesh.translate(-mesh.get_center()) # De-mean
    mesh_transform(mesh, Trafo3d(t=[0, 0, 350], rpy=np.deg2rad([180, 0.2, -0.3])))
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    if True:
        # Visualize scene
        cs = cam.get_cs(size=50.0)
        frustum = cam.get_frustum(size=300.0)
        o3d.visualization.draw_geometries([cs, frustum, mesh])

    if False:
        #cam.set_lighting_mode('parallel')
        #cam.set_light_vector((0, 0, 1))
        tic = time.monotonic()
        depth_image, color_image, pcl = cam.snap(mesh)
        toc = time.monotonic()
        print(f'Snapping image took {(toc - tic):.1f}s')
        show_images(depth_image, color_image)

    if True:
        lighting_modes = (
            'point',
            'point',
            'point',
            'parallel',
            'parallel',
            'parallel',
        )
        light_vectors = np.array((
            # point light positions
            (-100, 0, 0),
            (0, 0, 0),
            (100, 0, 0),
            # light directions
            (0, 0, 1),
            (0, 1, 1),
            (0, 2, 1),
            ))
        use_colormap = (True, True, True, False, False, False)
        cam.set_lighting_mode('point')
        fig = plt.figure()
        for i in range(6):
            cam.set_lighting_mode(lighting_modes[i])
            cam.set_light_vector(light_vectors[i])
            tic = time.monotonic()
            depth_image, color_image, pcl = cam.snap(mesh)
            toc = time.monotonic()
            print(f'Snapping image {i+1}/6 took {(toc - tic):.1f}s')
            ax = fig.add_subplot(2, 3, i+1)
            if use_colormap[i]:
                ax.imshow(color_image[:,:,0], cmap=cm.gray)
            else:
                ax.imshow(color_image)
            ax.set_axis_off()
            ax.set_aspect('equal')
            ax.set_title(f'mode "{lighting_modes[i]}", vec {light_vectors[i]}')

plt.show()
