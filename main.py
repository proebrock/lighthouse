import numpy as np
from trafolib.Trafo3d import Trafo3d
from CameraModel import CameraModel
from MeshObject import MeshObject
import matplotlib.pyplot as plt
plt.close('all')
import time




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
	phi_min = np.deg2rad(80)
	phi_max = np.deg2rad(90)
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



mesh = MeshObject()

if False:
    mesh.load('data/pyramid.ply')
    #mesh.load('data/knot.ply')
    #mesh.load('data/cube.ply')
    mesh.demean()
    mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
else:
    #mesh.generateFromImageFile('data/tux.png', 2.0)
    mesh.generateChArUco((3,2), 50.0)
    mesh.demean()
    mesh.transform(Trafo3d(rpy=np.deg2rad([155,25,0])))
#    mesh.transform(Trafo3d(rpy=np.deg2rad([180,0,0])))

#mesh.show(True, False, False)
	
generate_calibration_views(mesh, 5)

if False:
    cam = CameraModel((100, 100), 200, T=Trafo3d(t=(0,0,-500)))
    tic = time.process_time()
    dImg, cImg, P = cam.snap(mesh)
    toc = time.process_time()
    print(f'Snapping image took {(toc - tic):.1f}s')
    # Color of invalid pixels
    nan_color = (0, 0, 1.0)
    idx = np.where(np.isnan(cImg))
    cImg[idx[0], idx[1], :] = nan_color
    fig = plt.figure()

    ax = fig.add_subplot(121)
    cmap = plt.cm.viridis
    cmap.set_bad(color=nan_color, alpha=1.0)
    im = ax.imshow(1.0-dImg.T, cmap=cmap)
    #fig.colorbar(im, ax=ax)
    ax.set_axis_off()
    ax.set_title('Depth')
    ax.set_aspect('equal')

    ax = fig.add_subplot(122)
    ax.imshow(np.transpose(cImg, axes=(1,0,2)))
    ax.set_axis_off()
    ax.set_title('Color')
    ax.set_aspect('equal')

    plt.show()

