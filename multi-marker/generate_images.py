import numpy as np
import open3d as o3d
import json
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from camsimlib.camera_model import CameraModel
from camsimlib.o3d_utils import mesh_generate_plane, \
    mesh_generate_aruco_marker, save_shot
from trafolib.trafo3d import Trafo3d



if __name__ == "__main__":
     # Random but reproducible
    np.random.seed(42)
    # Path where to store the data
    data_dir = 'data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    print(f'Using data path "{data_dir}"')

    # Generate camera
    cam = CameraModel(chip_size=(40, 30),
                      focal_length=(35, 40),
                      distortion=(0.4, -0.2, 0, 0, 0))
    cam.scale_resolution(30)
    world_to_cam = Trafo3d()
    cam.set_pose(world_to_cam)

    # Generate object
    world_to_object = Trafo3d(t=(-30, 20, 580), rpy=np.deg2rad((170, -20, 20)))
    object = mesh_generate_plane((300, 200), color=(1, 1, 0))
    # Put coordinate system in center of plane
    object.translate((-150, -100, 0))
    # Move plane slightly in negative Z; if we shot images of it, the markers will be above the plane
    object.translate((0, 0, -1e-2))
    object.transform(world_to_object.get_homogeneous_matrix())

    # Generate markers
    marker_ids = np.array([0, 1, 2 ,3])
    marker_square_length = 30.0
    object_to_markers = (
        Trafo3d(t=(-140, -90, 0)),
        Trafo3d(t=(110, -90, 0)),
        Trafo3d(t=(-140, 60, 0)),
        Trafo3d(t=(110, 60, 0))
    )
    # Quote from Aruco documentation:
    # "For each marker, its four corners are returned in their original order
    # (which is clockwise starting with top left)"
    marker_points = marker_square_length * np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0], [0, 0, 0]])
    markers = []
    for i in range(len(marker_ids)):
        marker = mesh_generate_aruco_marker(marker_square_length, marker_ids[i])
        world_to_marker = world_to_object * object_to_markers[i]
        marker.transform(world_to_marker.get_homogeneous_matrix())
        markers.append(marker)

    # Visualize
    if False:
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=300.0)
        object_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        object_cs.transform(world_to_object.get_homogeneous_matrix())

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer('Open3D', 1024, 768)
        vis.show_settings = True
        vis.add_geometry('Camera CS', cam_cs)
        vis.add_geometry('Camera Frustum', cam_frustum)
        vis.add_geometry('Object CS', object_cs)
        vis.add_geometry('Object', object)
        for i in range(len(marker_ids)):
            vis.add_geometry(f'Marker {i}', markers[i])
#            vis.add_3d_label((marker_coords[i, 0],
#                              marker_coords[i, 1] + 20,
#                              marker_coords[i, 2]), f'Marker {i}')
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

    # Snap scene
    mesh = object
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
    pcl.transform(cam.get_pose().inverse().get_homogeneous_matrix())
    save_shot(basename, depth_image, color_image, pcl)

    # Save all scene
    params = {}
    params['cam'] = {}
    cam.dict_save(params['cam'])
    params['world_to_object'] = {}
    params['world_to_object']['t'] = world_to_object.get_translation().tolist()
    params['world_to_object']['q'] = world_to_object.get_rotation_quaternion().tolist()
    params['markers'] = {}
    for i in range(len(marker_ids)):
        params['markers'][i] = {}
        params['markers'][i]['object_to_marker'] = {}
        params['markers'][i]['object_to_marker']['t'] = object_to_markers[i].get_translation().tolist()
        params['markers'][i]['object_to_marker']['q'] = object_to_markers[i].get_rotation_quaternion().tolist()
        params['markers'][i]['square_length'] = marker_square_length
        params['markers'][i]['points'] = marker_points.tolist()
    with open(basename + '.json', 'w') as f:
       json.dump(params, f, indent=4, sort_keys=True)
