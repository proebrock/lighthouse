import cv2
import glob
import json
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
plt.close('all')
import open3d as o3d
from scipy.optimize import least_squares

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_generate_rays



def detect_circles(image, verbose=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_NONE)
    circles = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.8 * gray.size:
            # contour with 80 or more percent of total image size
            continue
        if True:
            # Result based on center of gravity and area->radius
            M = cv2.moments(c)
            circle = np.array([M["m10"] / M["m00"],
                               M["m01"] / M["m00"],
                               np.sqrt(area/np.pi)])
        else:
            # Result based on minimum enclosing circle
            circ = cv2.minEnclosingCircle(c)
            circle = np.array([circ[0][0], circ[0][1], circ[1]])
        circles.append((circle, np.asarray(c)[:,0,:]))
    if verbose:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for circle, contour in circles:
            ax.plot(*circle[0:2], 'r+')
            if False:
                # Plot circle
                circle_artist = plt.Circle(circle[0:2], circle[2],
                                            color='r', fill=False)
                ax.add_artist(circle_artist)
            else:
                # Plot contour points
                ax.plot(contour[:,0], contour[:,1], '.r')
        ax.set_aspect('equal')
        plt.show()
    return circles



def estimate_sphere_center_coarse(cam, circle_center, circle_radius, sphere_radius):
    f = np.mean(cam.get_focal_length())
    p = np.array([[
            circle_center[0], circle_center[1],
            # Ignoring all more advanced camera model parameters,
            # just using the naive focal length equation:
            # z / sphere_radius = f / circle_radius
            (f * sphere_radius) / circle_radius
            ]])
    sphere_center = cam.chip_to_scene(p)[0]
    return sphere_center



def ray_point_distance(rayorig, raydirs, point):
    """ Calculate minimal/orthogonal distance between ray and point
    If point projected to *line* defined by (rayorig + t * raydir) is on the *ray*
    (means that t>=0), return the distance to the line; otherwise return distance to rayorig
    """
    # Special case for single camera (one rayorig, multiple raydirs) and single point
    assert rayorig.ndim == 1 and rayorig.size == 3
    assert raydirs.ndim == 2 and raydirs.shape[1] == 3
    assert point.ndim == 1 and point.size == 3
    # rayorig vectors have to have length 1.0
    assert np.all(np.isclose(np.linalg.norm(raydirs, axis=1), 1.0))

    v = -rayorig + point
    t = np.sum(v * raydirs, axis=1) # dot product of v and raydir
    # P is point projected onto line defined by rayorig + t * raydir
    P = np.zeros_like(raydirs)
    mask = t>=0
    P[mask, :] = raydirs[mask, :] * t[mask, np.newaxis] + rayorig
    P[~mask, :] = rayorig
    # Distance between P and point
    dist = np.sqrt(np.sum(np.square(P - point), axis=1))
    return dist



def estimate_sphere_center_objfun(x, rayorig, raydirs, sphere_radius):
    return np.square(ray_point_distance(rayorig, raydirs, x) - sphere_radius)



def get_camera_rays(cam, circle_contour):
    """ Determine rays from camera to the circle_contour points
    """
    rayorig = cam.get_camera_pose().get_translation()
    p = np.zeros((circle_contour.shape[0], 3))
    p[:, 0:2] = circle_contour
    p[:, 2] = 1000.0
    P = cam.chip_to_scene(p)
    raydirs = P - rayorig
    raydirs /= np.linalg.norm(raydirs, axis=1)[:, np.newaxis]
    return rayorig, raydirs



def estimate_sphere_center(cam, circle_center, circle_radius, circle_contour, sphere_radius):
    x0 = estimate_sphere_center_coarse(cam, circle_center, circle_radius, sphere_radius)
    if True:
        rayorig, raydirs = get_camera_rays(cam, circle_contour)
        res = least_squares(estimate_sphere_center_objfun, x0,
                            args=(rayorig, raydirs, sphere_radius), verbose=0)
        if not res.success:
            raise Exception(f'Sphere : {res}')
        sphere_center = res.x
        return sphere_center
    else:
        # No optimization, just return the initial estimate
        return x0




def generate_sphere(sphere_center, sphere_radius):
    """ Generate sphere mesh for visualization
    """
    sphere = o3d.io.read_triangle_mesh('../data/sphere.ply')
    sphere.compute_triangle_normals()
    sphere.compute_vertex_normals()
    sphere.scale(sphere_radius, center=sphere.get_center())
    sphere.translate(-sphere.get_center())
    sphere.translate(sphere_center)
    sphere.paint_uniform_color((0.1, 0.5, 0.3))
    return sphere



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    #data_dir = 'a'
    data_dir = '/home/phil/pCloudSync/data/lighthouse/2d_ball_locate'
    if not os.path.exists(data_dir):
        raise Exception('Source directory does not exist.')

    # Load camera
    with open(os.path.join(data_dir, 'image00.json'), 'r') as f:
        params = json.load(f)
    cam = CameraModel()
    cam.dict_load(params['cam'])
    # Load images
    img_filenames = sorted(glob.glob(os.path.join(data_dir, '*_color.png')))
    images = []
    for filename in img_filenames:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    # Load real sphere positions
    config_filenames = sorted(glob.glob(os.path.join(data_dir, '*.json')))
    sphere_centers = np.zeros((len(images), 3))
    sphere_radius = None
    for i, filename in enumerate(config_filenames):
        with open(filename, 'r') as f:
            params = json.load(f)
        sphere_centers[i,:] = params['sphere']['center']
        sphere_radius = params['sphere']['radius']

    # Run circle detection on images
    circles = []
    for i, img in enumerate(images):
        circ = detect_circles(img, i == 23)
        if len(circ) != 1:
            raise Exception(f'Number of detected circles in image {img_filenames[i]} is {len(circ)}')
        circles.append(circ[0])

    # Visualize one instance
    if True:
        index = 23
        circle = circles[index]
        circle_center = circle[0][0:2]
        circle_radius = circle[0][2]
        circle_contour = circle[1]
        # Generate rays
        rayorig, raydirs = get_camera_rays(cam, circle_contour)
        t = 1100 # Length of rays
        rays = mesh_generate_rays(rayorig, rayorig + t * raydirs, (0,0,0))
        # Visualize
        cs = cam.get_cs(size=200)
        sphere = generate_sphere(sphere_centers[index, :], sphere_radius)
        o3d.visualization.draw_geometries([cs, sphere, rays])

    estimated_sphere_centers = np.zeros((len(images), 3))
    for i in range(len(images)):
        circle = circles[i]
        circle_center = circle[0][0:2]
        circle_radius = circle[0][2]
        circle_contour = circle[1]
        estimated_sphere_centers[i, :] = estimate_sphere_center(cam, \
            circle_center, circle_radius, circle_contour, sphere_radius)

    # Analysis
    errors = estimated_sphere_centers - sphere_centers
    abs_errors = np.linalg.norm(errors, axis=1)

    if False:
        fig, ax = plt.subplots()
        ax.set_title('Error per coordinate')
        ax.boxplot(errors)
        ax.set_xticklabels(['X', 'Y', 'Z'])
        ax.set_xlabel('Coordinate')
        ax.set_ylabel('Distance of estimated and real 3d sphere positions (mm)')
        ax.yaxis.grid(True)

    if True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(sphere_centers[:,2], errors[:,0], 'or', label='X')
        ax.plot(sphere_centers[:,2], errors[:,1], 'og', label='Y')
        ax.plot(sphere_centers[:,2], errors[:,2], 'ob', label='Z')
        ax.plot(sphere_centers[:,2], abs_errors, 'oc', label='dist')
        ax.legend(loc='best', fancybox=True, framealpha=0.5)
        ax.set_xlabel('Real z distance of sphere from camera (mm)')
        ax.set_ylabel('Error (mm)')
        ax.grid()
        plt.show()
