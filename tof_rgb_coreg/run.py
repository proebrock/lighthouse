import copy
import cv2
import json
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
plt.close('all')

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d


def load_scene(data_dir, scene_no):
    # Load ToF camera
    basename = os.path.join(data_dir, f'cam00_image{scene_no:02d}')
    with open(os.path.join(basename + '.json'), 'r') as f:
        params = json.load(f)
    tof_cam = CameraModel()
    tof_cam.dict_load(params['cam'])

    # Load ToF camera data
    pcl = o3d.io.read_point_cloud(basename + '.ply')
    # Convert color to gray
    colors = np.asarray(pcl.colors)
    grays = 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]
    grays = np.tile(grays, (3, 1)).T
    pcl.colors = o3d.utility.Vector3dVector(grays)
    # Transform points back from camera CS to world CS
    pcl.transform(tof_cam.get_camera_pose().get_homogeneous_matrix())

    # Load RGB camera
    basename = os.path.join(data_dir, f'cam01_image{scene_no:02d}')
    with open(os.path.join(basename + '.json'), 'r') as f:
        params = json.load(f)
    rgb_cam = CameraModel()
    rgb_cam.dict_load(params['cam'])

    # Load RGB camera data
    rgb_img = cv2.imread(basename + '_color.png')
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    return tof_cam, pcl, rgb_cam, rgb_img



def visualize_with_normals(pcl):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcl)
    vis.get_render_option().point_show_normal = True
    vis.run()
    vis.destroy_window()



def get_invalid_chip_coordinates_mask(rgb_cam, p):
    return np.logical_or.reduce((
        p[:,0] < 0,
        p[:,0] > rgb_cam.get_chip_size()[0],
        p[:,1] < 0,
        p[:,1] > rgb_cam.get_chip_size()[1],
    ))



def get_invalid_view_direction_mask(rgb_cam, pcl):
    assert np.asarray(pcl.normals).shape[0] > 0 # Point cloud must contain normals
    # Get view direction of RGB camera to point cloud points
    view_dirs = -np.asarray(pcl.points) + rgb_cam.get_camera_pose().get_translation()
    view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1)[:,np.newaxis]
    # Calculate angles between normal vector and view direction to rgb camera
    normals = np.asarray(pcl.normals)
    angles = np.arccos(np.sum(view_dirs * normals, axis=1)) # Dot product
    angles = np.rad2deg(angles)
    return angles > 90.0



def interpolate_gray_image(img, p):
    assert img.ndim == 2
    assert p.ndim == 2
    assert p.shape[1] == 2
    # Generate pixel coordinates
    x = np.arange(img.shape[1])
    y = np.arange(img.shape[0])
    x, y = np.meshgrid(x, y)
    # Interpolate
    img_unit = img.astype(float).ravel() / 255.0
    return griddata((x.ravel(), y.ravel()), img_unit, p, method='linear')



def interpolate_rgb_image(img, p):
    assert img.ndim == 3
    assert img.shape[2] == 3 # 3 channel color
    assert p.ndim == 2
    assert p.shape[1] == 2
    colors = np.empty((p.shape[0], 3))
    # Separate interpolations for each color channel
    colors[:, 0] = interpolate_gray_image(img[:, :, 0], p)
    colors[:, 1] = interpolate_gray_image(img[:, :, 1], p)
    colors[:, 2] = interpolate_gray_image(img[:, :, 2], p)
    return colors



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/tof_rgb_coreg'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    tof_cam, pcl, rgb_cam, rgb_img = load_scene(data_dir, 0)

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
    pcl.orient_normals_towards_camera_location(tof_cam.get_camera_pose().get_translation())
    #visualize_with_normals(pcl)

    # Transform points to RGB camera chip
    p = rgb_cam.scene_to_chip(np.asarray(pcl.points))
    p = p[:,0:2] # Omit distance value
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img)
        ax.plot(p[:,0], p[:,1], 'rx')
        plt.show()

    # Mask pixels we cannot determine colors for
    invalid_chip_coords = get_invalid_chip_coordinates_mask(rgb_cam, p)
    print(f'Invalid chip coordinates: {sum(invalid_chip_coords)}')
    invalid_view_dir = get_invalid_view_direction_mask(rgb_cam, pcl)
    print(f'Invalid view directions: {sum(invalid_view_dir)}')
    invalid_mask = np.logical_or(invalid_chip_coords, invalid_view_dir)

    # For each point in p determine the color by interpolating over the camera chip
    colors = np.empty(np.asarray(pcl.points).shape)
    colors[~invalid_mask, :] = interpolate_rgb_image(rgb_img, p[~invalid_mask, :])
    colors[invalid_chip_coords, :] = (1, 0, 0)
    colors[invalid_view_dir, :] = (0, 0, 1)
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
