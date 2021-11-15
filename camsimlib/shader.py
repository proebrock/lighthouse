import numpy as np



class Shader:

    def __init__(self, mesh, light_type='point', light_vector=(0, 0, 0)):
        self.mesh = mesh
        self.light_type = light_type
        if not light_type in ['point', 'parallel']:
            raise Exception(f'Unknown light type {light_type}')
        self.light_vector = np.asarray(light_vector)
        if light_vector.ndim != 1 or light_vector.size != 3:
            raise Exception(f'Invalid light vector {light_vector}')



    def run(self, ray_tracer):
        # Extract vertices and vertex normals from mesh
        triangle_idx = ray_tracer.get_triangle_indices() # shape (n, )
        triangles = np.asarray(self.mesh.triangles)[triangle_idx, :] # shape (n, 3)
        vertices = np.asarray(self.mesh.vertices)[triangles] # shape (n, 3, 3)
        vertex_normals = np.asarray(self.mesh.vertex_normals)[triangles] # shape (n, 3, 3)



