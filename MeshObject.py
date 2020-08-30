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


	def __str__(self):
		return \
			f'min {np.min(self.vertices, axis=0)}\n' + \
			f'max {np.max(self.vertices, axis=0)}\n' + \
			f'range {np.max(self.vertices, axis=0)-np.min(self.vertices, axis=0)}\n'


	def load(self, filename):
		self.mesh = o3d.io.read_triangle_mesh(filename)
		if not self.mesh.has_triangles():
			raise Exception('Triangle mesh expected.')
		self.__extractMesh()



	def demean(self):
		self.mesh.translate(-np.mean(np.asarray(self.mesh.vertices), axis=0))
		self.__extractMesh()



	def transform(self, T):
		self.mesh.rotate(T.GetRotationMatrix(), center=(0,0,0))
		self.mesh.translate(T.GetTranslation())
		self.__extractMesh()



	def __extractMesh(self):
		if not self.mesh.has_triangle_normals():
			self.mesh.compute_triangle_normals()
		if not self.mesh.has_vertex_normals():
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

