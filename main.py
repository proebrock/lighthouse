import open3d as o3d
import numpy as np
from cameraModel import CameraModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


# Load mesh
if True:
	mesh = o3d.io.read_triangle_mesh('data/cube.ply')
	for m in mesh.vertices:
		m *= 100.0
else:
	mesh = o3d.io.read_triangle_mesh('data/knot.ply')
# Transform: de-mean
mesh.translate(-np.mean(np.asarray(mesh.vertices), axis=0))
# Transform: rotate
R = mesh.get_rotation_matrix_from_xyz((np.pi/4,np.pi/4,0))
mesh.rotate(R, center=(0,0,0))
# Transform: translate; camera view direction is z axis
mesh.translate(( 0.0, 0.0, 500.0 ))
# Compute norrmals
mesh.compute_triangle_normals()
mesh.compute_vertex_normals()
print(mesh)

# Show mesh
if True:
	cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=100.0, origin=[ 0.0, 0.0, 0.0 ])
	o3d.visualization.draw_geometries([mesh, cs], point_show_normal=True)

if False:
	vertices = np.asarray(mesh.vertices)
	triangles = np.asarray(mesh.triangles)
	triangle_normals = np.asarray(mesh.triangle_normals)
	# vertices
	i = 0
	print(vertices.shape)
	print(vertices[i,:])
	print('min', np.min(vertices, axis=0))
	print('max', np.max(vertices, axis=0))
	# triangles
	print(f'has triangles: {mesh.has_triangles()}')
	print(triangles.shape)
	print(triangles[i,:])
	print(vertices[triangles[i,:]].shape)
	# triangle normals
	print(f'has triangle normals: {mesh.has_triangle_normals()}')
	print(triangle_normals.shape)
	print(triangle_normals[i,:])
	# calculate normal and compare with original
	e0 = vertices[triangles[i,0]]
	e1 = vertices[triangles[i,1]]
	e2 = vertices[triangles[i,2]]
	n = np.cross(e1-e0, e2-e0)
	n = n / np.linalg.norm(n)
	print(n)
	# vertex colors
	print(f'has vertex colors: {mesh.has_vertex_colors()}')

if True:
	# Create camera and snap image
	width = 100
	height = 100
	cam = CameraModel((width, height), (200, 200))
	tic = time.process_time()
	dImg, cImg = cam.snap(mesh)
	toc = time.process_time()
	print(f'Snapping image took {(toc - tic):.1f}s')

	fig = plt.figure()
	ax = fig.add_subplot(121)
	ax.imshow(cImg)
	ax.set_axis_off()
	ax.set_title('Color')
	ax.set_aspect('equal')
	ax = fig.add_subplot(122)
	ax.imshow(dImg)
	ax.set_axis_off()
	ax.set_title('Depth')
	ax.set_aspect('equal')
	plt.show()

