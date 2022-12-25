import numpy as np
import open3d as o3d



class MultiMesh:

    def __init__(self, meshes=None, mirrors=None):
        pass



    def __str__(self):
        pass



    def isclose(self, other):
        pass



    def num_vertices(self):
        return self.vertices.shape[0]



    def num_triangles(self):
        return self.triangles.shape[0]



    def triangle_vertices():
        pass



    def clear(self):
        self.vertices = np.array((0, 3))
        self.vertex_normals = np.array((0, 3))
        self.vertex_colors = np.array((0, 3))
        self.triangles = np.array((0, 3), dtype=int)
        self.triangle_normals = np.array((0, 3))
        self.num_meshes = 0
        self.mesh_indices = np.array(0, dtype=int)
        self.is_mirror = np.array(0, dtype=bool)



    def from_o3d_mesh(self, mesh, mirror=None):
        self.vertices = np.asarray(mesh.vertices)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.triangles = np.asarray(mesh.triangles)
        self.triangle_normals = np.asarray(mesh.triangle_normals)
        self.num_meshes = 1
        self.mesh_indices = np.zeros(self.num_triangles(), dtype=int)



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
            mask = (self.mesh_indices == i)
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(self.vertices[mask, :])
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals[mask, :])
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors[mask, :])
            mesh.triangles = o3d.utility.Vector3iVector(self.triangles[mask])
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals[mask, :])
            meshes.append(mesh)
        return meshes



    def to_o3d_tensor_mesh_list(self):
        pass



    def show(self):
        pass # Use o3d to show
