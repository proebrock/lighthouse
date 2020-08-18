import open3d as o3d
import numpy as np
from cameraModel import CameraModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import time



# Load mesh and transform
if True:
	mesh = o3d.io.read_triangle_mesh('data/cube.ply')
	for m in mesh.vertices:
		m *= 100.0
else:
	mesh = o3d.io.read_triangle_mesh('data/knot.ply')
mesh.compute_vertex_normals()
offset = np.array([ 0.0, 0.0, 500.0 ]) - np.mean(np.asarray(mesh.vertices), axis=0)
for m in mesh.vertices:
	m += offset
print(mesh)

# Show mesh
if False:
	cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
		size=100.0, origin=[ 0.0, 0.0, 0.0 ])
	o3d.visualization.draw_geometries([mesh, cs])


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
	print(triangles.shape)
	print(triangles[i,:])
	print(vertices[triangles[i,:]].shape)
	# triangle normals
	print(triangle_normals.shape)
	print(triangle_normals[i,:])
	# calculate normal and compare with original
	e0 = vertices[triangles[i,0]]
	e1 = vertices[triangles[i,1]]
	e2 = vertices[triangles[i,2]]
	n = np.cross(e1-e0, e2-e0)
	n = n / np.linalg.norm(n)
	print(n)

# Create camera and snap image
width = 100
height = 100
cam = CameraModel((width, height), (200, 200))
tic = time.process_time()
P = cam.snap(mesh)
toc = time.process_time()
print(f'Snapping image took {(toc - tic):.1f}s')
img = cam.scenePointsToDepthImage(P)

# Display image
fig = plt.figure()
ax = fig.add_subplot(111)
sns.heatmap(img.T, ax=ax, cmap=sns.cm.rocket_r)
ax.set_aspect('equal')
plt.show()

