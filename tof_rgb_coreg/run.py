import cv2
import json
import numpy as np
import open3d as o3d
import os
import sys
import matplotlib.pyplot as plt
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



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    #data_dir = '/home/phil/pCloudSync/data/lighthouse/tof_rgb_coreg'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    tof_cam, pcl, rgb_cam, rgb_img = load_scene(data_dir, 0)

    # Estimate normal vectors for point cloud
    pcl.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30),
        fast_normal_computation=False)
    pcl.orient_normals_towards_camera_location(tof_cam.get_camera_pose().get_translation())
    visualize_with_normals(pcl)

    if False:
        # Get view direction of RGB camera to point cloud points
        view_dirs = -np.asarray(pcl.points) + rgb_cam.get_camera_pose().get_translation()
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=1)[:,np.newaxis]
        # Calculate angles between normal vector and view direction to rgb camera
        normals = np.asarray(pcl.normals)
        angles = np.arccos(np.sum(view_dirs * normals, axis=1)) # Dot product
        angles = np.rad2deg(angles)

