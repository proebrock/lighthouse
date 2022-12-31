import numpy as np
import open3d as o3d



class MultiMesh:

    def __init__(self, meshes=None, mirrors=None):
        if meshes is None and mirrors is None:
            self.clear()
        elif isinstance(meshes, o3d.cpu.pybind.geometry.TriangleMesh):
            if mirrors is None:
                self.from_o3d_mesh(meshes, False)
            else:
                self.from_o3d_mesh(meshes, mirrors)
        elif isinstance(meshes, list) or isinstance(meshes, tuple):
            if len(meshes) == 0:
                self.clear()
            elif isinstance(meshes[0], o3d.cpu.pybind.geometry.TriangleMesh):
                self.from_o3d_mesh_list(meshes, mirrors)
        else:
            raise ValueError('Unknown type provided')



    def num_vertices(self):
        return self.vertices.shape[0]



    def num_triangles(self):
        return self.triangles.shape[0]



    def has_vertex_colors(self):
        return self.vertex_colors.shape[0] > 0



    def clear(self):
        num_vertices = 0
        self.vertices = np.zeros((num_vertices, 3))
        self.vertex_normals = np.zeros((num_vertices, 3))
        self.vertex_colors = np.zeros((num_vertices, 3))
        num_triangles = 0
        self.triangles = np.zeros((num_triangles, 3), dtype=int)
        self.triangle_normals = np.zeros((num_triangles, 3))
        self.triangle_is_mirror = np.zeros(num_triangles, dtype=bool)



    def from_o3d_mesh(self, mesh, mirror=False):
        # Make sure mesh has normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        # Transfer data
        self.vertices = np.asarray(mesh.vertices)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.triangles = np.asarray(mesh.triangles)
        self.triangle_normals = np.asarray(mesh.triangle_normals)
        self.triangle_is_mirror = mirror * np.ones(self.num_triangles(), dtype=bool)



    def from_o3d_mesh_list(self, meshes, mirrors=None):
        if mirrors is None:
            mirrors = np.zeros(len(meshes), dtype=bool)
        elif len(meshes) != len(mirrors):
            raise ValueError('Invalid number of meshes and mirror flags provided.')
        self.triangle_is_mirror = np.zeros(0, dtype=bool)
        combined_mesh = o3d.geometry.TriangleMesh()
        for i, mesh in enumerate(meshes):
            # Make sure mesh has normals
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            # Join all meshes into a single one
            combined_mesh += mesh
            # Generate mirror flags for each triangle
            num_new_triangles = np.asarray(mesh.triangles).shape[0]
            self.triangle_is_mirror = np.hstack((self.triangle_is_mirror,
                mirrors[i] * np.ones(num_new_triangles, dtype=bool)))
        # Transfer data
        self.vertices = np.asarray(combined_mesh.vertices)
        self.vertex_normals = np.asarray(combined_mesh.vertex_normals)
        self.vertex_colors = np.asarray(combined_mesh.vertex_colors)
        self.triangles = np.asarray(combined_mesh.triangles)
        self.triangle_normals = np.asarray(combined_mesh.triangle_normals)



    def from_ply_filename(self, filename, mirror=False):
        mesh = o3d.io.read_triangle_mesh(filename)
        self.from_o3d_mesh(mesh, mirror)



    def from_ply_filename_list(self, filenames, mirrors=None):
        meshes = []
        for filename in filenames:
            meshes.append(o3d.io.read_triangle_mesh(filename))
        self.from_o3d_mesh_list(meshes, mirrors)



    def from_components_lists(self, vertices_list, triangles_list, mirrors=None):
        if len(vertices_list) != len(triangles_list):
            raise ValueError('Provide proper lists of components')
        self.clear()
        meshes = []
        for vertices, triangles in zip(vertices_list, triangles_list):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector( \
                np.asarray(vertices))
            mesh.triangles = o3d.utility.Vector3iVector( \
                np.asarray(triangles, dtype=int))
            meshes.append(mesh)
        self.from_o3d_mesh_list(meshes, mirrors)



    def to_o3d_mesh(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals)
        return mesh



    def to_o3d_tensor_mesh(self):
        mesh = self.to_o3d_mesh()
        return o3d.t.geometry.TriangleMesh.from_legacy(mesh)
