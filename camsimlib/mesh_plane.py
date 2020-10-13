import numpy as np

from . mesh_object import MeshObject



class MeshPlane(MeshObject):

    def __init__(self, shape, color=(1,1,1)):
        # Call base class constructor
        super(MeshPlane, self).__init__()
        self.vertices = np.array([
                [0, 0, 0],
                [shape[0], 0, 0],
                [shape[0], shape[1], 0],
                [0, shape[1], 0]])
        self.vertex_normals = np.zeros((4, 3))
        self.vertex_normals[:, 2] = 1.0
        self.vertex_colors = np.array([color, color, color, color])
        self.triangles = np.array([[0, 1, 3], [2, 3, 1]])
        self.triangle_normals = np.zeros((2, 3))
        self.triangle_normals[:, 2] = 1.0
        self.triangle_vertices = self.vertices[self.triangles]
        self.numpy_to_o3d()
