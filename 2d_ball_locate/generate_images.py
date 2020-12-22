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
from camsimlib.o3d_utils import show_images, save_shot



def object_is_fully_in_view(cam, mesh, verbose=False):
    # Generate a camera with a chip 3x3 times as big
    hicam = copy.deepcopy(cam)
    hicam.set_chip_size(3 * hicam.get_chip_size())
    hicam.set_principal_point(3 * hicam.get_principal_point())
    # Snap image
    depth_image, color_image, pcl = hicam.snap(mesh)
    # Analyze image
    w, h = cam.get_chip_size()
    # The original camera image is in the center of the 3x3
    inner_image = depth_image[h:2*h,w:2*w]
    # Number of valid pixels in the inner image
    good_pixel_count = np.sum(~np.isnan(inner_image))
    # Number of valid pixels in the outer frame
    bad_pixel_count = np.sum(~np.isnan(depth_image)) - good_pixel_count
    # Debug output
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(depth_image)
        ax.axvline(x=w, color='r')
        ax.axvline(x=2*w, color='r')
        ax.axhline(y=h, color='r')
        ax.axhline(y=2*h, color='r')
        ax.text(0, 0, f'bad:{bad_pixel_count}', va='top', size=10, color='k')
        ax.text(w, h, f'good:{good_pixel_count}', va='top', size=10, color='k')
        plt.show()
    # We want all pixels to be in the inner image and no outside
    return (good_pixel_count > 0) and (bad_pixel_count == 0)



def visualize_max_distance(cam, sphere):
    mesh = copy.deepcopy(sphere)
    mesh.translate((0, 0, 4000))
    depth_image, color_image, pcl = snap_cam.snap(mesh)
    show_images(depth_image, color_image)



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
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Setup cameras
    cam = CameraModel(chip_size=(80, 60),
                      focal_length=(80, 88))
    # Camera for taking the final image
    snap_cam = copy.deepcopy(cam)
    snap_cam.scale_resolution(15)

    # Create sphere
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere_radius = 40.0
    sphere.scale(sphere_radius / 50.0, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.paint_uniform_color((0.1, 0.5, 0.3))

#    visualize_max_distance(cam, sphere)
#    visualize_scene(cam, sphere, 100)

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
        # Check if fully in view
        print(f'    Snapping test image...')
        if not object_is_fully_in_view(cam, mesh, verbose=False):
            print('    Sphere not fully in view, re-generating ...')
            num_failed += 1
            continue
        # Take final image
        print(f'    Snapping final image ...')
        tic = time.process_time()
        depth_image, color_image, pcl = snap_cam.snap(mesh)
        toc = time.process_time()
        print(f'    Snapping took {(toc - tic):.1f}s.')
        # Save generated snap
        basename = os.path.join(data_dir, f'image{img_no:02d}')
        # Save PCL in camera coodinate system, not in world coordinate system
        pcl.transform(snap_cam.get_camera_pose().inverse().get_homogeneous_matrix())
        save_shot(basename, depth_image, color_image, pcl)
        # Save all image parameters
        params = {}
        params['cam'] = {}
        snap_cam.dict_save(params['cam'])
        params['sphere'] = {}
        params['sphere']['center'] = sphere_center.tolist()
        params['sphere']['radius'] = sphere_radius
        with open(basename + '.json', 'w') as f:
           json.dump(params, f, indent=4, sort_keys=True)
        # Check if we are done
        img_no = img_no + 1
        if img_no >= num_images:
            break;
    print(f'Done ({num_failed} failed).')
