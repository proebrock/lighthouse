import numpy as np
import open3d as o3d
import os
import sys
import time

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
                      focal_length=(40, 45),
                      distortion=(-0.8, 0.8, 0, 0, 0))
    cam.scale_resolution(4)
    cam.place_camera((230, 10, 450))
    cam.look_at((120, 90, 0))
    cam.roll_camera(np.deg2rad(25))

    # Generate plane with markers
    plane = mesh_generate_plane((300, 200), color=(1, 1, 0))
    eps = 1e-2 # Z-distance of marker above the X/Y plane
    marker_coords = np.array([ \
        [10, 10], [260, 10], [10, 160], [260, 160]])
    marker_square_length = 30.0
    markers = []
    for marker_id, coords in enumerate(marker_coords):
        marker = mesh_generate_aruco_marker(marker_square_length, marker_id)
        marker.translate((coords[0], coords[1], eps))
        markers.append(marker)


    if True:
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=300.0)
        plane_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)

#        o3d.visualization.draw_geometries([cam_cs, cam_frustum, plane_cs, mesh])

        app = o3d.visualization.gui.Application.instance
        app.initialize()
        vis = o3d.visualization.O3DVisualizer('Open3D', 1024, 768)
        vis.show_settings = True
        vis.add_geometry('Camera CS', cam_cs)
        vis.add_geometry('Camera Frustum', cam_frustum)
        vis.add_geometry('Plane CS', plane_cs)
        vis.add_geometry('Plane', plane)
        for i in range(len(markers)):
            vis.add_geometry(f'Marker {i}', markers[i])
            vis.add_3d_label((marker_coords[i, 0],
                              marker_coords[i, 1], 0), f'Marker {i}')
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()

#    # Snap scene
#    scene = plane
#    for marker in markers:
#        scene = scene + marker
#    basename = os.path.join(data_dir, f'cam00_image00')
#    print(f'Snapping image {basename} ...')
#    tic = time.process_time()
#    depth_image, color_image, pcl = cam.snap(mesh)
#    toc = time.process_time()
#    print(f'    Snapping image took {(toc - tic):.1f}s')
#    # Save generated snap
#    # Save PCL in camera coodinate system, not in world coordinate system
#    pcl.transform(cam.get_camera_pose().inverse().get_homogeneous_matrix())
#    save_shot(basename, depth_image, color_image, pcl)