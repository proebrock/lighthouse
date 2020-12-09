import glob
import json
import numpy as np
import os
import sys
import open3d as o3d
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from trafolib.trafo3d import Trafo3d



def load_cams(data_dir):
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    cams = []
    for filename in filenames:
        print(f'Loading camera from {filename} ...')
        with open(filename) as f:
            params = json.load(f)
        cam = CameraModel()
        cam.dict_load(params['cam'])
        cams.append(cam)
    return cams



def load_circle_params(data_dir):
    filename = os.path.join(data_dir, 'cam00_image00.json')
    with open(filename) as f:
        params = json.load(f)
    sphere_center = np.array(params['sphere']['center'])
    sphere_radius = params['sphere']['radius']
    return sphere_center, sphere_radius



def load_pcl(data_dir, cams):
    # Load point clouds from all cameras
    filenames = sorted(glob.glob(os.path.join(data_dir, '*.ply')))
    clouds = []
    for i, filename in enumerate(filenames):
        pcl = o3d.io.read_point_cloud(filename)
        pcl.transform(cams[i].get_camera_pose().get_homogeneous_matrix())
        clouds.append(pcl)
    return clouds



def visualize_scene(cams, clouds):
    scene = [ ]
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), # RGB
              (0, 1, 1), (1, 0, 1), (1, 1, 0)) # CMY
    for i, cloud in enumerate(clouds):
        pcl = o3d.geometry.PointCloud(cloud)
        pcl.paint_uniform_color(colors[i])
        scene.append(pcl)
#    for cam in cams:
#        scene.append(cam.get_cs(size=100.0))
    o3d.visualization.draw_geometries(scene)



def sphere_from_four_points(points):
    """ Find equation of sphere from 4 points
    :param points: Numpy array of size 4x3
    """
    # We calculate the 4x4 determinant by dividing the problem in determinants of 3x3 matrix
    # Multiplied by (x²+y²+z²)
    d_matrix = np.ones((4, 4))
    for i in range(4):
        d_matrix[i, 0] = points[i, 0]
        d_matrix[i, 1] = points[i, 1]
        d_matrix[i, 2] = points[i, 2]
    M11 = np.linalg.det(d_matrix)
    # Multiplied by x
    for i in range(4):
        d_matrix[i, 0] = np.dot(points[i], points[i])
        d_matrix[i, 1] = points[i, 1]
        d_matrix[i, 2] = points[i, 2]
    M12 = np.linalg.det(d_matrix)
    # Multiplied by y
    for i in range(4):
        d_matrix[i, 0] = np.dot(points[i], points[i])
        d_matrix[i, 1] = points[i, 0]
        d_matrix[i, 2] = points[i, 2]
    M13 = np.linalg.det(d_matrix)
    # Multiplied by z
    for i in range(4):
        d_matrix[i, 0] = np.dot(points[i], points[i])
        d_matrix[i, 1] = points[i, 0]
        d_matrix[i, 2] = points[i, 1]
    M14 = np.linalg.det(d_matrix)
    # Multiplied by 1
    for i in range(4):
        d_matrix[i, 0] = np.dot(points[i], points[i])
        d_matrix[i, 1] = points[i, 0]
        d_matrix[i, 2] = points[i, 1]
        d_matrix[i, 3] = points[i, 2]
    M15 = np.linalg.det(d_matrix)
    # Now we calculate the center and radius
    center = np.array([0.5*(M12/M11), -0.5*(M13/M11), 0.5*(M14/M11)])
    radius = np.sqrt(np.dot(center, center) - (M15 / M11))
    return center, radius



def fit_sphere_objfun(x, points, radius):
    dist_from_center = np.linalg.norm(points - x, axis=1)
    return dist_from_center - radius



def fit_sphere_local_optimize(points, radius):
    # To get a good start point for local optimization, we randomly
    # pick four points and estimate the sphere center as a start point
    # for further numerical optimization
    n = points.shape[0]
    samples = np.random.choice(range(n), 4, replace=False)
    x0, radius = sphere_from_four_points(points[samples,:])
#    with np.printoptions(precision=2):
#        print(f'Starting point for numerical optimization: x0={x0} (radius={radius:.2f})')
    res = least_squares(fit_sphere_objfun, x0, xtol=0.1,
                        args=(points, radius))
    if not res.success:
        raise Exception(f'Bundle adjustment failed: {res}')
    return res.x, np.sqrt(np.sum(np.square(res.fun)) / n)



def fit_sphere(points, radius, num_start_points=20):
    # Global optimization
    centers = np.zeros((num_start_points, 3))
    errors = np.zeros(num_start_points)
    for i in range(num_start_points):
        centers[i,:], errors[i] = fit_sphere_local_optimize(points, radius)
    best_index = np.argmin(errors)
    return centers[best_index,:], errors[best_index]



if __name__ == "__main__":
    # Config
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load cameras and point clouds
    cameras = load_cams(data_dir)
    sphere_center, sphere_radius = load_circle_params(data_dir)
    clouds = load_pcl(data_dir, cameras)
    # Join all clouds
    allclouds = o3d.geometry.PointCloud()
    for cloud in clouds:
        allclouds += cloud
    points = np.asarray(allclouds.points)
    visualize_scene(cameras, clouds)

    # Run sphere fitting
    print('\nRunning sphere fitting ...')
    estimated_sphere_center, residuals = fit_sphere(points, sphere_radius)
    print(f'Real sphere center at {sphere_center}')
    print(f'Estimated sphere center at {estimated_sphere_center}')
    print(f'Error {estimated_sphere_center - sphere_center}')
    print(f'Residuals per cam {residuals:.3f}')

    # Add small error to camera pose of one camera
    T_small = Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 1, 0)))
    cam_no = 1
    cameras[cam_no].set_camera_pose(cameras[cam_no].get_camera_pose() * T_small)
    clouds = load_pcl(data_dir, cameras)
    # Join all clouds
    allclouds = o3d.geometry.PointCloud()
    for cloud in clouds:
        allclouds += cloud
    points = np.asarray(allclouds.points)
    visualize_scene(cameras, clouds)

    # Re-run bundle adjustment
    print(f'\nRe-running sphere fitting after misaligning camera {cam_no}...')
    estimated_sphere_center, residuals = fit_sphere(points, sphere_radius)
    print(f'Real sphere center at {sphere_center}')
    print(f'Estimated sphere center at {estimated_sphere_center}')
    print(f'Error {estimated_sphere_center - sphere_center}')
    print(f'Residuals per cam {residuals:.3f}')

