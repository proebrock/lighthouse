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
        pcl.transform(cams[i].get_pose().get_homogeneous_matrix())
        clouds.append(np.asarray(pcl.points))
    return clouds



def visualize_scene(cams, clouds):
    scene = []
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), # RGB
              (0, 1, 1), (1, 0, 1), (1, 1, 0)) # CMY
    for i, cloud in enumerate(clouds):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(cloud)
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



def compress_residuals(residuals, clouds):
    num_cams = len(clouds)
    num_points_per_cam = np.array(list(clouds[i].shape[0] for i in range(num_cams)))
    residuals_per_cam = np.zeros(num_cams)
    k = 0
    for i, j in enumerate(np.cumsum(num_points_per_cam)):
        residuals_per_cam[i] = np.sqrt(np.sum(np.square(residuals[k:j])) / clouds[i].shape[0])
        k = j
    return residuals_per_cam



def fit_sphere_local_optimize(clouds, radius):
    # Combine clouds in single array
    allclouds = np.zeros((0,3))
    for cloud in clouds:
        allclouds = np.vstack((allclouds, cloud))
    # To get a good start point for local optimization, we randomly
    # pick four points and estimate the sphere center as a start point
    # for further numerical optimization
    n = allclouds.shape[0]
    samples = np.random.choice(range(n), 4, replace=False)
    x0, radius = sphere_from_four_points(allclouds[samples,:])
#    with np.printoptions(precision=1, suppress=True):
#        print(f'Starting point for numerical optimization: x0={x0} (radius={radius:.2f})')
    res = least_squares(fit_sphere_objfun, x0, xtol=0.1,
                        args=(allclouds, radius))
    if not res.success:
        raise Exception(f'Bundle adjustment failed: {res}')
    return res.x, compress_residuals(res.fun, clouds)



def fit_sphere(clouds, radius, num_start_points=20):
    # Global optimization
    centers = np.zeros((num_start_points, 3))
    errors = np.zeros((num_start_points, len(clouds)))
    for i in range(num_start_points):
        centers[i,:], errors[i, :] = fit_sphere_local_optimize(clouds, radius)
    best_index = np.argmin(np.sum(errors, axis=1))
    return centers[best_index,:], errors[best_index,:]



def fit_sphere_sac(clouds, radius, num_start_points=20, threshold=1.0, verbose=False):
    # Combine clouds in single array
    allclouds = np.zeros((0,3))
    for cloud in clouds:
        allclouds = np.vstack((allclouds, cloud))
    # Sample consensus approach
    residuals = np.empty((len(clouds), len(clouds)))
    for i, cloud in enumerate(clouds):
        # Run sphere fitting on a single cloud from a single camera
        # and then calculate residuals for all cameras
        x, _ = fit_sphere((cloud), radius, num_start_points)
        fun = fit_sphere_objfun(x, allclouds, radius)
        residuals[i,:] = compress_residuals(fun, clouds)
    # For each solution from above determine number of inlier cameras
    # by checking residuals against a maximum theshold (in pixels)
    all_inliers = residuals < threshold
    # Use row with maximum number of inliers
    max_inliers_index = np.argmax(np.sum(all_inliers, axis=1))
    # This is a bool vector determining the inliers of cameras
    cam_inliers = all_inliers[max_inliers_index, :]
    if verbose:
        with np.printoptions(precision=1, suppress=True):
            print('residuals\n', residuals)
            print('all_inliers\n', all_inliers)
            print('max_inliers_index\n', max_inliers_index)
            print('cam_inliers\n', cam_inliers)
    # Run sphere fit with all inlier cameras to determine result
    good_clouds = list(clouds[i] for i in np.where(cam_inliers)[0])
    x, _ = fit_sphere(good_clouds, radius, num_start_points)
    # Use this solution from inlier cameras and determine residuals for ALL cams
    fun = fit_sphere_objfun(x, allclouds, radius)
    return x, compress_residuals(fun, clouds), cam_inliers



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    # Config
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/bundle_adjust_simple'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load cameras and point clouds
    cameras = load_cams(data_dir)
    sphere_center, sphere_radius = load_circle_params(data_dir)
    clouds = load_pcl(data_dir, cameras)
    visualize_scene(cameras, clouds)

    # Run sphere fitting
    print('\nRunning sphere fitting ...')
    estimated_sphere_center, residuals = fit_sphere(clouds, sphere_radius)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Residuals per cam {residuals} mm')

    # Add small error to camera pose of one camera
    T_small = Trafo3d(t=(0, 0, 0), rpy=np.deg2rad((0, 1, 0)))
    cam_no = 1
    cameras[cam_no].set_pose(cameras[cam_no].get_pose() * T_small)
    clouds = load_pcl(data_dir, cameras)
    visualize_scene(cameras, clouds)

    # Re-run bundle adjustment
    print(f'\nRe-running sphere fitting after misaligning camera {cam_no}...')
    estimated_sphere_center, residuals = fit_sphere(clouds, sphere_radius)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Residuals per cam {residuals} mm')

    # Run bundle adjustment with sample consensus (SAC) approach
    threshold = 1.0
    print(f'\nRe-running SAC sphere fitting (threshold={threshold} mm) after misaligning camera {cam_no}...')
    estimated_sphere_center, residuals, cam_inliers = \
        fit_sphere_sac(clouds, sphere_radius, threshold=threshold, verbose=False)
    with np.printoptions(precision=1, suppress=True):
        print(f'Real sphere center at {sphere_center} mm')
        print(f'Estimated sphere center at {estimated_sphere_center} mm')
        print(f'Error {estimated_sphere_center - sphere_center} mm')
        print(f'Residuals per cam {residuals} mm')
        print(f'Inlier cameras: {cam_inliers}')
