import numpy as np
import open3d as o3d



class MultiMesh:

    def __init__(self, meshes=None, mirrors=None):
        pass



    def __str__(self):
        pass



    def num_vertices(self):
        return self.vertices.shape[0]



    def num_triangles(self):
        return self.triangles.shape[0]



    def triangle_vertices():
        pass



    def clear(self, num_vertices=0, num_triangles=0):
        self.vertices = np.zeros((num_vertices, 3))
        self.vertex_normals = np.zeros((num_vertices, 3))
        self.vertex_colors = np.zeros((num_vertices, 3))
        self.vertex_mesh = np.zeros(num_vertices, dtype=int)
        self.triangles = np.zeros((num_triangles, 3), dtype=int)
        self.triangle_normals = np.zeros((num_triangles, 3))
        self.triangle_mesh = np.zeros(num_triangles, dtype=int)
        self.num_meshes = 0
        self.is_mirror = np.zeros(self.num_meshes, dtype=bool)



    def from_o3d_mesh(self, mesh, mirror=False):
        self.vertices = np.asarray(mesh.vertices)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.vertex_mesh = np.zeros(self.num_triangles(), dtype=int)
        self.triangles = np.asarray(mesh.triangles)
        self.triangle_normals = np.asarray(mesh.triangle_normals)
        self.triangle_mesh = np.zeros(self.num_triangles(), dtype=int)
        self.num_meshes = 1
        self.is_mirror = np.array((mirror), dtype=bool)



    def from_o3d_mesh_list(self, meshes, mirrors=None):
        pass



    def from_ply_filenames(self, filenames, mirrors=None):
        pass



    def to_o3d_mesh(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals)
        return mesh



    def to_o3d_mesh_list(self):
        meshes = []
        for i in range(self.num_meshes):
            mesh = o3d.geometry.TriangleMesh()
            vertex_mask = (self.vertex_mesh == i)
            mesh.vertices = o3d.utility.Vector3dVector(self.vertices[vertex_mask, :])
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals[vertex_mask, :])
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors[vertex_mask, :])
            triangle_mask = (self.triangle_mesh == i)
            mesh.triangles = o3d.utility.Vector3iVector(self.triangles[triangle_mask])
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals[triangle_mask, :])
            meshes.append(mesh)
        return meshes



    def to_o3d_tensor_mesh_list(self):
        pass



    def show(self):
        pass # Use o3d to show
