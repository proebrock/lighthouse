import numpy as np
import open3d as o3d



class MultiMesh:

    def __init__(self, meshes=None, mirrors=None):
        if meshes is None:
            self.clear()
        elif isinstance(meshes, o3d.cpu.pybind.geometry.TriangleMesh):
            self.from_o3d_mesh(meshes, mirrors)
        elif isinstance(meshes, str):
            self.from_ply_filename(meshes, mirrors)
        elif isinstance(meshes, list) or isinstance(meshes, tuple):
            if len(meshes) == 0:
                self.clear()
            elif isinstance(meshes[0], o3d.cpu.pybind.geometry.TriangleMesh):
                self.from_o3d_mesh_list(meshes, mirrors)
            elif isinstance(meshes[0], str):
                self.from_ply_filenames(meshes, mirrors)
        else:
            self.clear()



    def __str__(self):
        pass



    def num_vertices(self):
        return self.vertices.shape[0]



    def num_triangles(self):
        return self.triangles.shape[0]



    def num_meshes(self):
        return self.is_mirror.size



    def triangle_vertices():
        pass



    def clear(self):
        num_vertices = 0
        self.vertices = np.zeros((num_vertices, 3))
        self.vertex_normals = np.zeros((num_vertices, 3))
        self.vertex_colors = np.zeros((num_vertices, 3))
        self.vertex_mesh = np.zeros(num_vertices, dtype=int)
        num_triangles = 0
        self.triangles = np.zeros((num_triangles, 3), dtype=int)
        self.triangle_normals = np.zeros((num_triangles, 3))
        self.triangle_mesh = np.zeros(num_triangles, dtype=int)
        num_meshes = 0
        self.is_mirror = np.zeros(num_meshes, dtype=bool)



    def from_o3d_mesh(self, mesh, mirror=False):
        # Make sure mesh has normals
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        self.vertices = np.asarray(mesh.vertices)
        self.vertex_normals = np.asarray(mesh.vertex_normals)
        if mesh.has_vertex_colors():
            self.vertex_colors = np.asarray(mesh.vertex_colors)
        self.vertex_mesh = np.zeros(self.num_vertices(), dtype=int)
        self.triangles = np.asarray(mesh.triangles)
        self.triangle_normals = np.asarray(mesh.triangle_normals)
        self.triangle_mesh = np.zeros(self.num_triangles(), dtype=int)
        self.is_mirror = np.array((mirror), dtype=bool)



    def from_o3d_mesh_list(self, meshes, mirrors=None):
        if len(meshes) == 0:
            self.clear()
            return
        # Make sure each mesh has normals
        for mesh in meshes:
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
        # Determine sizes needed and pre-allocate memory
        total_num_vertices = 0
        total_num_triangles = 0
        for mesh in meshes:
            num_vertices  += np.asarray(mesh.vertices).shape[0]
            num_triangles += np.asarray(mesh.triangles).shape[0]
        # Allocate memory
        self.clear(total_num_vertices, total_num_triangles, len(meshes))
        # Transfer data
        vstart = 0
        tstart = 0
        for i, mesh in enumerate(meshes):
            vend = vstart + np.asarray(mesh.vertices).shape[0]
            tend = tstart + np.asarray(mesh.triangles).shape[0]
            self.vertices[vstart:vend, :] = np.asarray(mesh.vertices)
            self.vertex_normals[vstart:vend, :] = np.asarray(mesh.vertex_normals)
            self.vertex_colors[vstart:vend, :] = np.asarray(mesh.vertex_colors)
            self.vertex_mesh[vstart:vend] = i
            self.triangles[tstart:tend, :] = np.asarray(mesh.triangles)
            self.triangle_normals[tstart:tend, :] = np.asarray(mesh.triangle_normals)
            self.triangle_mesh[tstart:tend] = i
            vstart = vend
            tstart = tend
        # Handle mirror flags
        if mirrors is None:
            self.is_mirror[:] = False
        else:
            if len(meshes) != len(mirrors):
                raise ValueError('Invalid number of meshes and mirror flags provided.')
            self.is_mirror = np.asarray(mirrors, dtype=bool)



    def from_ply_filename(self, filename, mirror=False):
        mesh = o3d.io.read_triangle_mesh(filename)
        self.from_o3d_mesh(mesh, mirror)



    def from_ply_filenames(self, filenames, mirrors=None):
        meshes = []
        for filename in filenames:
            meshes.append(o3d.io.read_triangle_mesh(filename))
        self.from_o3d_mesh_list(meshes, mirrors)



    def from_components(self, vertices, triangles):
        self.clear()
        self.vertices = np.asarray(vertices)
        self.vertex_mesh = np.zeros(self.num_vertices(), dtype=int)
        self.triangles = np.asarray(triangles)
        self.triangle_mesh = np.zeros(self.num_triangles(), dtype=int)
        self.is_mirror = np.array((False), dtype=bool)



    def to_o3d_mesh(self):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        if self.vertex_normals.size > 0: # Optional
            mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        if self.vertex_colors.size > 0: # Optional
            mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        if self.triangle_normals.size > 0: # Optional
            mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals)
        return mesh, self.is_mirror



    def to_o3d_mesh_list(self):
        meshes = []
        for i in range(self.num_meshes()):
            mesh = o3d.geometry.TriangleMesh()
            vertex_mask = (self.vertex_mesh == i)
            mesh.vertices = o3d.utility.Vector3dVector( \
                self.vertices[vertex_mask, :])
            if self.vertex_normals.size > 0: # Optional
                mesh.vertex_normals = o3d.utility.Vector3dVector( \
                    self.vertex_normals[vertex_mask, :])
            if self.vertex_colors.size > 0: # Optional
                mesh.vertex_colors = o3d.utility.Vector3dVector(\
                    self.vertex_colors[vertex_mask, :])
            triangle_mask = (self.triangle_mesh == i)
            mesh.triangles = o3d.utility.Vector3iVector( \
                self.triangles[triangle_mask])
            if self.triangle_normals.size > 0: # Optional
                mesh.triangle_normals = o3d.utility.Vector3dVector( \
                    self.triangle_normals[triangle_mask, :])
            meshes.append(mesh)
        return meshes, self.is_mirror



    def to_o3d_tensor_mesh_list(self):
        meshes, mirrors = self.to_o3d_mesh_list()
        meshes = list(o3d.t.geometry.TriangleMesh.from_legacy(mesh) for mesh in meshes)
        return meshes, mirrors



    def show(self, cs_size=-1):
        objects = self.to_o3d_mesh_list()
        if cs_size < 0:
            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=cs_size)
            objects.append(cs)
        o3d.visualization.draw_geometries(objects)
