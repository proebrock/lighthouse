import copy
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from common.image_utils import image_load
from common.mesh_utils import pcl_load
from camsimlib.camera_model import CameraModel
from camsimlib.image_mapping import image_sample_points_bilinear



def load_scene(data_dir, scene_no):
    # Load ToF camera
    basename = os.path.join(data_dir, f'cam00_image{scene_no:02d}')
    with open(os.path.join(basename + '.json'), 'r') as f:
        params = json.load(f)
    tof_cam = CameraModel()
    tof_cam.dict_load(params['cam'])

    # Load ToF camera data
    pcl = pcl_load(basename + '.ply')
    # Transform points back from camera CS to world CS
    pcl.transform(tof_cam.get_pose().get_homogeneous_matrix())
    # Ground truth: Colored point cloud from ToF camera
    colored_pcl_orig = copy.deepcopy(pcl)
    # Convert color to gray
    colors = np.asarray(pcl.colors)
    grays = 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]
    grays = np.tile(grays, (3, 1)).T
    pcl.colors = o3d.utility.Vector3dVector(grays)

    # Load RGB camera
    basename = os.path.join(data_dir, f'cam01_image{scene_no:02d}')
    with open(os.path.join(basename + '.json'), 'r') as f:
        params = json.load(f)
    rgb_cam = CameraModel()
    rgb_cam.dict_load(params['cam'])

    # Load RGB camera data
    rgb_img = image_load(basename + '.png')

    return tof_cam, colored_pcl_orig, pcl, rgb_cam, rgb_img



def visualize_with_normals(pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcl)
    vis.get_render_option().point_show_normal = True
    vis.run()
    vis.destroy_window()



def get_invalid_view_direction_mask(rgb_cam, pcl):
    assert np.asarray(pcl.normals).shape[0] > 0 # Point cloud must contain normals
    # Get view direction of RGB camera to point cloud points
    view_dirs = -np.asarray(pcl.points) + rgb_cam.get_pose().get_translation()
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1)[:,np.newaxis]
    # Calculate angles between normal vector and view direction to rgb camera
    normals = np.asarray(pcl.normals)
    angles = np.arccos(np.sum(view_dirs * normals, axis=1)) # Dot product
    angles = np.rad2deg(angles)
    if True:
        pcl_view = copy.deepcopy(pcl)
        colorize_point_cloud_by_scalar(pcl_view, angles)
        o3d.visualization.draw_geometries([pcl_view])
    return angles > 90.0



def colorize_point_cloud_by_scalar(pcl, values, min_max=None, nan_color=(1, 0, 0)):
    assert values.ndim == 1
    assert np.asarray(pcl.points).shape[0] == values.size
    cm = plt.get_cmap('jet')
    #cm = plt.get_cmap('viridis')
    isvalid = ~np.isnan(values)
    if min_max is None:
        min_max = (np.min(values[isvalid]), np.max(values[isvalid]))
    values_norm = np.clip((values[isvalid] - min_max[0]) / (min_max[1] - min_max[0]), 0, 1)
    colors = np.empty((np.asarray(pcl.points).shape[0], 3))
    colors[isvalid, :] = cm(values_norm)[:, 0:3]
    colors[~isvalid, :] = nan_color
    pcl.colors = o3d.utility.Vector3dVector(colors)



if __name__ == "__main__":
    # Random but reproducible
    np.random.seed(42)
    # Get data path
    data_path_env_var = 'LIGHTHOUSE_DATA_DIR'
    if data_path_env_var in os.environ:
        data_dir = os.environ[data_path_env_var]
        data_dir = os.path.join(data_dir, 'tof_rgb_coreg')
    else:
        data_dir = 'data'
    data_dir = os.path.abspath(data_dir)
    print(f'Using data from "{data_dir}"')

    tof_cam, colored_pcl_orig, pcl, rgb_cam, rgb_img = load_scene(data_dir, 0)
    if False:
        o3d.visualization.draw_geometries([pcl])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img)
        ax.set_aspect('equal')
        plt.show()

    # Comment on invalid pixels:
    # When the ToF camera does not receive enough light on some pixels because the
    # object is too far away, this is detected as an invalid pixel and removed from the
    # point cloud. So the number of points in pcl is smaller or equal the number of
    # pixels of the ToF camera.
    # The RGB camera does not have active lighting, it just records the incoming light
    # on its pixels, so there are no "invalid" pixels. So we have a full image, the
    # background pixels are filled with the color "cyan" (0, 1, 1).
    # Dependent on the application, it would be possible to place an object against
    # a black background and then segment the image and extract the object. This would
    # give you a mask of "valid" pixels. This information may be used for some algorithms
    # but we stick here to the common case of having a full RGB image.

    # Estimate normal vectors for point cloud
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
        fast_normal_computation=False)
    pcl.orient_normals_towards_camera_location(tof_cam.get_pose().get_translation())
    #visualize_with_normals(pcl)

    # Transform points to RGB camera chip
    p = rgb_cam.scene_to_chip(np.asarray(pcl.points))
    p = p[:,0:2] # Omit distance value
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img)
        ax.plot(p[:,0], p[:,1], 'rx')
        ax.set_aspect('equal')
        plt.show()

    # Mask pixels we cannot determine colors for
    print(f'Total number of points: {p.shape[0]}')
    invalid_chip_coords = ~rgb_cam.points_on_chip_mask(p)
    print(f'Invalid chip coordinates: {sum(invalid_chip_coords)} ({sum(invalid_chip_coords)*100.0/p.shape[0]:.1f}%)')
    invalid_view_dir = get_invalid_view_direction_mask(rgb_cam, pcl)
    print(f'Invalid view directions: {sum(invalid_view_dir)} ({sum(invalid_view_dir)*100.0/p.shape[0]:.1f}%)')
    invalid_mask = np.logical_or(invalid_chip_coords, invalid_view_dir)
    print(f'Valid points: {np.sum(~invalid_mask)} ({sum(~invalid_mask)*100.0/p.shape[0]:.1f}%)')

    # For each point in p determine the color by interpolating over the camera chip
    colors = np.empty(np.asarray(pcl.points).shape)
    samples, _ = image_sample_points_bilinear(rgb_img, p[~invalid_mask, :])
    colors[~invalid_mask, :] = samples / 255.0
    colors[invalid_chip_coords, :] = (1, 0, 0) # Red
    colors[invalid_view_dir, :] = (0, 1, 0) # Green
    colored_pcl = copy.deepcopy(pcl)
    colored_pcl.colors = o3d.utility.Vector3dVector(colors)

    # Visualize result
    if True:
        tof_cam_cs = tof_cam.get_cs(size=50.0)
        tof_cam_frustum = tof_cam.get_frustum(size=300.0)
        rgb_cam_cs = rgb_cam.get_cs(size=100.0) # RGB has bigger coordinate system
        rgb_cam_frustum = rgb_cam.get_frustum(size=300.0)
        o3d.visualization.draw_geometries([tof_cam_cs, tof_cam_frustum,
            rgb_cam_cs, rgb_cam_frustum, colored_pcl])

    # Display quality of reconstructed colors
    if True:
        colors = np.asarray(colored_pcl.colors)
        colors_orig = np.asarray(colored_pcl_orig.colors)
        d = np.sqrt(np.sum(np.square(colors - colors_orig), axis=1))
        pcl_quality = copy.deepcopy(pcl)
        colorize_point_cloud_by_scalar(pcl_quality, d, min_max=(0, 0.5), nan_color=(0, 1, 1))
        o3d.visualization.draw_geometries([pcl_quality])
