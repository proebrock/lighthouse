import glob
import json
import numpy as np
import os
import sys
import open3d as o3d

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d




if __name__ == "__main__":
    # Config
    data_dir = 'a'

    # Load cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    cameras = []
    sphere_radius = None
    for filename in filenames:
        print(f'Loading camera from {filename} ...')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cameras.append(cam)
        sphere_center = np.array(params['sphere']['center'])
        sphere_radius = params['sphere']['radius']

    # Load point clouds from all cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.ply')))
    cloud = o3d.geometry.PointCloud()
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), # RGB
              (0, 1, 1), (1, 0, 1), (1, 1, 0)) # CMY
    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(filename)
        pcl.transform(cameras[i].get_camera_pose().get_homogeneous_matrix())
        pcl.paint_uniform_color(colors[i])
        cloud += pcl
    print(f'Total number of points is {np.asarray(cloud.points).size}')

#    cloud = cloud.voxel_down_sample(voxel_size=1.0)
#    print(f'After downsampling {np.asarray(cloud.points).size}')
    o3d.visualization.draw_geometries([cloud])

