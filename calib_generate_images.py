import numpy as np
from trafolib.Trafo3d import Trafo3d
from CameraModel import CameraModel
from CharucoBoard import CharucoBoard
import time
import cv2
import json



def generate_calibration_views(mesh, n_views):
    # We assume the calibration plate is in X/Y plane with Z=0
    mesh_min = np.min(mesh.vertices, axis=0)
    mesh_max = np.max(mesh.vertices, axis=0)
    # Pick point to look at on the surface of the mesh
    look_at_pos = np.zeros((n_views, 3))
    look_at_pos[:,0] = np.random.uniform(mesh_min[0], mesh_max[0], n_views)
    look_at_pos[:,1] = np.random.uniform(mesh_min[1], mesh_max[1], n_views)
    # Pick a camera position in X/Y
    cam_scale = 2.0
    camera_pos = np.zeros((n_views, 3))
    camera_pos[:,0] = np.random.uniform(cam_scale*mesh_min[0], cam_scale*mesh_max[0], n_views)
    camera_pos[:,1] = np.random.uniform(cam_scale*mesh_min[1], cam_scale*mesh_max[1], n_views)
    # Camera Z position is determined by desired view angle
    phi_min = np.deg2rad(30)
    phi_max = np.deg2rad(70)
    phi = np.random.uniform(phi_min, phi_max, n_views)
    camera_pos[:,2] = np.linalg.norm(look_at_pos - camera_pos, axis=1) * np.tan(phi)
    # Unit vector in Z is direction from camera to view point
    ez = look_at_pos - camera_pos
    ez /= np.linalg.norm(ez, axis=1).reshape(n_views,1)
    # Unit vector in Y is perpendicular to ez
    ey = np.ones((n_views, 3))
    ey[:,2] = (- ey[:,0] * ez[:,0] - ey[:,1] * ez[:,1]) / ez[:,2]
    ey /= np.linalg.norm(ey, axis=1).reshape(n_views,1)
    # Unit vector in X is perpendicular to ey and ez
    ex = np.cross(ey, ez, axis=1)
    # Assemble transformations
    trafos = []
    for i in range(n_views):
        R = np.vstack((ex[i,:], ey[i,:], ez[i,:])).T
        T = Trafo3d(t=camera_pos[i,:], mat=R)
        trafos.append(T)
    return trafos



def save_image(filename, img):
    # Find NaN values
    nanidx = np.where(np.isnan(img))
    # Convert image to integer
    img = (255.0 * img).astype(np.uint8)
    # Convert RGB to gray image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Maximize dynamic range
    img = ((img - np.min(img)) * 255.0) / (np.max(img) - np.min(img))
    # Set NaN to distinct color
    img[nanidx[0],nanidx[1]] = 127
    # Write image, transpose for OpenCV
    cv2.imwrite(filename, img.T)



np.random.seed(42)
board = CharucoBoard((8,10), 30.0)
board.show(True, False, False)
trafos = generate_calibration_views(board, 25)

for i, T in enumerate(trafos):
    print(f'Snapping image {i+1}/{len(trafos)} ...')
    a = 7 # Use this scale factor to control image size and computation time
    cam = CameraModel(pix_size=(120*a, 100*a), f=(70*a,75*a), c=(66*a,50*a),trafo=T)
    tic = time.process_time()
    dImg, cImg, P = cam.snap(board)
    toc = time.process_time()
    print(f'    Snapping image took {(toc - tic):.1f}s')
    # Save image
    save_image(f'image{i:02d}.png', cImg)
    # Save all image parameters
    params = {}
    params['cam'] = {}
    cam.dict_save(params['cam'])
    params['board'] = {}
    board.dict_save(params['board'])
    with open(f'image{i:02d}.json', 'w') as f:
       json.dump(params, f, indent=4, sort_keys=True)

print('Done.')
