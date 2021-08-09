import numpy as np
import open3d as o3d
import json
import os
import sys
import time
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_generate_plane, \
    mesh_generate_aruco_marker, save_shot
from trafolib.trafo3d import Trafo3d



if __name__ == "__main__":
    np.random.seed(42) # Random but reproducible
    data_dir = 'a'
    if not os.path.exists(data_dir):
        raise Exception('Target directory does not exist.')

    # Generate camera
    cam = CameraModel(chip_size=(40, 30),
                      focal_length=(35, 40),
                      distortion=(0.4, -0.2, 0, 0, 0))
    cam.scale_resolution(50)
    cam.place_camera((230, 10, 550))
    cam.look_at((120, 90, 0))
    cam.roll_camera(np.deg2rad(25))

    # Generate plane
    world_to_plane = Trafo3d()
    plane = mesh_generate_plane((300, 200), color=(1, 1, 0))
    plane.transform(world_to_plane.get_homogeneous_matrix())
    # Generate markers
    eps = 1e-2 # Z-distance of marker above the X/Y plane
    marker_ids = np.array([0, 1, 2 ,3])
    marker_coords = np.array([ \
        [10, 10, eps], [260, 10, eps], [10, 160, eps], [260, 160, eps]])
    marker_square_length = 30.0
    # Quote from Aruco documentation:
    # "For each marker, its four corners are returned in their original order
    # (which is clockwise starting with top left)"
    marker_points = marker_square_length * np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
    markers = []
    for i in range(len(marker_ids)):
        marker = mesh_generate_aruco_marker(marker_square_length, marker_ids[i])
        marker.translate(marker_coords[i,:])
        markers.append(marker)

    # Visualize
    if False:
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=300.0)
        plane_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer('Open3D', 1024, 768)
        vis.show_settings = True
        vis.add_geometry('Camera CS', cam_cs)
        vis.add_geometry('Camera Frustum', cam_frustum)
        vis.add_geometry('Plane CS', plane_cs)
        vis.add_geometry('Plane', plane)
        for i in range(len(marker_ids)):
            vis.add_geometry(f'Marker {i}', markers[i])
            vis.add_3d_label((marker_coords[i, 0],
                              marker_coords[i, 1] + 20,
                              marker_coords[i, 2]), f'Marker {i}')
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

    # Snap scene
    mesh = plane
    for marker in markers:
        mesh = mesh + marker
    basename = os.path.join(data_dir, f'cam00_image00')
    print(f'Snapping image {basename} ...')
    tic = time.monotonic()
    depth_image, color_image, pcl = cam.snap(mesh)
    toc = time.monotonic()
    print(f'    Snapping image took {(toc - tic):.1f}s')

    # Save generated snap
    # Save PCL in camera coodinate system, not in world coordinate system
    pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
    save_shot(basename, depth_image, color_image, pcl)

    # Save all image parameters
    params = {}
    params['cam'] = {}
    cam.dict_save(params['cam'])
    params['plane_pose'] = {}
    params['plane_pose']['t'] = world_to_plane.get_translation().tolist()
    params['plane_pose']['q'] = world_to_plane.get_rotation_quaternion().tolist()
    params['markers'] = {}
    for i in range(len(marker_ids)):
        coords = marker_points + marker_coords[i, :]
        params['markers'][i] = { 'coords': coords.tolist(),
            'square_length': marker_square_length }
    with open(basename + '.json', 'w') as f:
       json.dump(params, f, indent=4, sort_keys=True)

    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i in range(len(marker_ids)):
            coords = marker_points + marker_coords[i, :]
            ax.plot(coords[:,0], coords[:,1], 'xk')
            m = np.mean(coords, axis=0)
            ax.text(m[0], m[1], f'Marker {i}',
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('Object points, Z=0')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.show()

