import numpy as np
import open3d as o3d
import json
import os
import sys
import time

sys.path.append(os.path.abspath('../'))
from trafolib.trafo3d import Trafo3d
from common.image_utils import image_3float_to_rgb, image_save
from common.mesh_utils import mesh_generate_plane
from common.aruco_utils import MultiAruco
from camsimlib.camera_model import CameraModel



def generate_object(pose):
    # Plane
    plane = mesh_generate_plane((300, 200), color=(1, 1, 0))
    # Put coordinate system in center of plane
    plane.translate((-150, -100, 0))
    # Move plane slightly in negative Z; if we shot images of it, the markers will be above the plane
    plane.translate((0, 0, -1e-2))
    plane.transform((pose * Trafo3d(rpy=(np.pi, 0, 0))).get_homogeneous_matrix())
    # Markers
    markers = MultiAruco(length_pix=30, length_mm=30, pose=pose)
    markers.add_marker(0, Trafo3d(t=(-140, -90, 0)))
    markers.add_marker(1, Trafo3d(t=( 110, -90, 0)))
    markers.add_marker(2, Trafo3d(t=(-140,  60, 0)))
    markers.add_marker(3, Trafo3d(t=( 110,  60, 0)))
    return plane + markers.generate_mesh(), markers



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

    # Generate object
    world_to_object = Trafo3d(t=(-30, 20, 580), rpy=np.deg2rad((-10, -20, 20)))
    mesh, markers = generate_object(world_to_object)

    # Visualize
    if False:
        cam_cs = cam.get_cs(size=50.0)
        cam_frustum = cam.get_frustum(size=300.0)
        object_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        object_cs.transform(world_to_object.get_homogeneous_matrix())
        o3d.visualization.draw_geometries([cam_cs, cam_frustum,
            object_cs, mesh])

    # Snap scene
    basename = os.path.join(data_dir, f'cam00_image00')
    print(f'Snapping image {basename} ...')
    tic = time.monotonic()
    _, image, _ = cam.snap(mesh)
    toc = time.monotonic()
    print(f'    Snapping image took {(toc - tic):.1f}s')
    # Save generated snap
    image = image_3float_to_rgb(image)
    image_save(basename + '.png', image)
    # Save parameters
    params = {}
    params['cam'] = {}
    cam.dict_save(params['cam'])
    params['markers'] = {}
    markers.dict_save(params['markers'])
    with open(basename + '.json', 'w') as f:
       json.dump(params, f, indent=4, sort_keys=True)

