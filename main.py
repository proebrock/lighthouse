import open3d as o3d
import numpy as np



def rayIntersectTriangle(rayDir, t):
	rayOrigin = np.array([0.0, 0.0, 0.0])
	v0 = t[0,:]
	v1 = t[1,:]
	v2 = t[2,:]
	e1 = v1 - v0
	e2 = v2 - v0
	h = np.cross(rayDir, e2)
	a = np.dot(e1, h)
	if np.isclose(a, 0.0):
		return None
	f = 1.0 / a
	s = rayOrigin - v0
	u = f * np.dot(s, h)
	if (u < 0.0) or (u > 1.0):
		return None
	q = np.cross(s, e1)
	v = f * np.dot(rayDir, q)
	if (v < 0.0) or ((u + v) > 1.0):
		return None
	t = f * np.dot(e2, q)
	if t <= 0.0:
		return None
	return rayOrigin + (rayDir * t)


if False:
	mesh = o3d.io.read_triangle_mesh('data/cube.ply')
	mesh.compute_vertex_normals()
	for m in mesh.vertices:
		m *= 100.0

if True:
	mesh = o3d.io.read_triangle_mesh('data/knot.ply')
	mesh.compute_vertex_normals()

offset = np.array([ 0.0, 0.0, 500.0 ]) - np.mean(np.asarray(mesh.vertices), axis=0)
for m in mesh.vertices:
	m += offset


vertices = np.asarray(mesh.vertices)
triangles = np.asarray(mesh.triangles)
triangle_normals = np.asarray(mesh.triangle_normals)

if False:
	print(mesh)
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
	# visualize
	cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=100.0, origin=[ 0.0, 0.0, 0.0 ])
	o3d.visualization.draw_geometries([mesh, cs])


ray = np.array([0, 0.0, 1.0])
for t in triangles:
	p = rayIntersectTriangle(ray, vertices[t])
	if p is not None:
		print(p)
