import pytest
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from common.mesh_utils import mesh_generate_image



def test_mesh_generate_image_coordinates():
    # Generate image
    h = 60
    w = 80
    m = 5
    image = np.zeros((h, w, 3), dtype=np.uint8) # All black
    image[0:m, 0:m, :] = (255, 255, 255) # Top-left:   white
    image[0:m, w-m:w, :] = (255, 0, 0) # Top-right:    red
    image[h-m:h, 0:m, :] = (0, 255, 0) # Bottom-left:  green
    image[h-m:h, w-m:w, :] = (0, 0, 255) # Bottom-right: blue
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(image)
        plt.show()
    # Generate mesh from image
    pixel_size = 0.1
    mesh = mesh_generate_image(image, pixel_size=pixel_size)
    if False:
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6)
        o3d.visualization.draw_geometries([cs, mesh])
    # Check color of mesh at certain coordinates
    ptree = o3d.geometry.KDTreeFlann(mesh)
    colors = np.asarray(mesh.vertex_colors)
    points = pixel_size * np.array((
        (w/2, h/2, 0), # Middle
        (0, 0, 0),     # Top-left
        (w, 0, 0),     # Top-right
        (0, h, 0),     # Bottom-left
        (w, h, 0),     # Bottom-right
    ))
    expected_colors = np.array((
        (0, 0, 0), # Black
        (1, 1, 1), # White
        (1, 0, 0), # Red
        (0, 1, 0), # Green
        (0, 0, 1), # Blue
    ))
    for p, ec in zip(points, expected_colors):
        [k, idx, _] = ptree.search_knn_vector_3d(p, 1)
        assert k ==  1
        assert np.all(colors[idx, :] == ec) # White
