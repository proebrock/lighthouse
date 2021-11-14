import numpy as np



class Shader:

    def __init__(self, mesh, light_type='auto'):
        self.mesh = mesh
        self.light_type = light_type # auto, point, parallel
        self.point_light_source_position = np.asarray(point_light_source_position)
        self.parallel_light_direction = np.asarray(parallel_light_direction)



    def run(self, triangle_idx, Pbary):
        # Extract vertices and vertex normals from mesh
        triangles = np.asarray(self.mesh.triangles)[triangle_idx, :]
        vertices = np.asarray(self.mesh.vertices)[triangles]
        vertex_normals = np.asarray(self.mesh.vertex_normals)[triangles]



