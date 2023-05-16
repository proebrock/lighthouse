import copy
import json
import numpy as np
import open3d as o3d
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from common.circle_detect import detect_circle_contours, detect_circle_hough
from common.image_utils import image_3float_to_rgb, image_save, image_show




def visualize_max_distance(cam, sphere):
    mesh = copy.deepcopy(sphere)
    mesh.translate((0, 0, 4000))
    _, image, _ = cam.snap(mesh)
    image = image_3float_to_rgb(image)
    image_show(image)



def visualize_scene(cam, sphere, num_spheres):
    objs = []
    objs.append(cam.get_cs(size=100.0))
    objs.append(cam.get_frustum(size=4000.0))
    for i in range(num_spheres):
        # Get new sphere center
        sphere_center = np.array([
                np.random.uniform(-1200, 1200),
                np.random.uniform(-900, 900),
                np.random.uniform(200, 4000)
                ])
        # Transform sphere
        mesh = copy.deepcopy(sphere)
        mesh.translate(sphere_center)
        objs.append(mesh)
    o3d.visualization.draw_geometries(objs)



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Setup camera
    cam = CameraModel(chip_size=(80, 60),
                      focal_length=(80, 88))
    cam.scale_resolution(15)

    # Create sphere
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere_radius = 40.0
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.paint_uniform_color((0.1, 0.5, 0.3))

    #visualize_max_distance(cam, sphere)
    #visualize_scene(cam, sphere, 100)

    num_images = 100
    num_failed = 0
    img_no = 0

    while True:
        print(f'Generating image {img_no+1}/{num_images}')
        # Get new sphere center
        sphere_center = np.array([
                np.random.uniform(-1200, 1200),
                np.random.uniform(-900, 900),
                np.random.uniform(200, 4000)
                ])
        # Transform sphere
        mesh = copy.deepcopy(sphere)
        mesh.translate(sphere_center)
        # Snap image
        print(f'    Snapping image ...')
        tic = time.monotonic()
        _, color_image, _ = cam.snap(mesh)
        toc = time.monotonic()
        print(f'    Snapping took {(toc - tic):.1f}s.')
        # Check if circle visible
        image = image_3float_to_rgb(color_image)
        circles, _ = detect_circle_contours(image, verbose=False)
        #circles = detect_circle_hough(image, verbose=False)
        if circles.shape[0] == 0:
            print('    # Circle detection failed.')
            num_failed += 1
            continue
        if  np.any(color_image[:, 0, :] > 0) or \
            np.any(color_image[:, -1, :] > 0) or \
            np.any(color_image[0, :, :] > 0) or \
            np.any(color_image[-1, :, :] > 0):
            print('    # Circle touches border of image.')
            num_failed += 1
            continue
        # Save generated snap
        basename = os.path.join(data_dir, f'image{img_no:02d}')
        # Save generated snap
        image_save(basename + '.png', image)
        # Save all image parameters
        params = {}
        params['cam'] = {}
        cam.dict_save(params['cam'])
        params['sphere'] = {}
        params['sphere']['center'] = sphere_center.tolist()
        params['sphere']['radius'] = sphere_radius
        with open(basename + '.json', 'w') as f:
           json.dump(params, f, indent=4, sort_keys=True)
        # Check if we are done
        img_no = img_no + 1
        if img_no >= num_images:
            break;
    print(f'Done ({num_failed} failed and had to be re-tried).')
