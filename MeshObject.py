import open3d as o3d
import numpy as np



class MeshObject:

	def __init__(self):
		self.mesh = None
		self.vertices = None
		self.vertex_normals = None
		self.vertex_colors = None
		self.triangles = None
		self.triangle_normals = None

	def load(self, filename, demean=False):
		self.mesh = o3d.io.read_triangle_mesh(filename)
		if not self.mesh.has_triangles():
			raise Exception('Triangle mesh expected.')
		if demean:
			self.mesh.translate(-np.mean(np.asarray(self.mesh.vertices), axis=0))

#		R = self.mesh.get_rotation_matrix_from_xyz((np.pi/4,np.pi/4,0))
#		self.mesh.rotate(R, center=(0,0,0))
#		self.mesh.translate(( 0.0, 0.0, 500.0 ))

		self.mesh.compute_triangle_normals()
		self.mesh.compute_vertex_normals()

		self.vertices = np.asarray(self.mesh.vertices)
		self.vertex_normals = np.asarray(self.mesh.vertex_normals)
		if self.mesh.has_vertex_colors():
			self.vertex_colors = np.asarray(self.mesh.vertex_colors)
		self.triangles = np.asarray(self.mesh.triangles)
		self.triangle_normals = np.asarray(self.mesh.triangle_normals)
		self.triangle_vertices = self.vertices[self.triangles]
	
	def show(self):
		# Coordinate system
		cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
			size=100.0, origin=[ 0.0, 0.0, 0.0 ])
		# Convert mesh to point cloud to visualize vertices and vertex normals
		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(self.vertices)
		pcd.normals = o3d.utility.Vector3dVector(self.vertex_normals)
		# Visualize
		o3d.visualization.draw_geometries([self.mesh, pcd, cs], point_show_normal=True)
