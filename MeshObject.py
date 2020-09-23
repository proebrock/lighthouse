import open3d as o3d
import cv2
import numpy as np



class MeshObject:

    def __init__(self):
        self.mesh = o3d.geometry.TriangleMesh()
        self.__o3d_to_numpy()



    def __str__(self):
        return \
            f'min {np.min(self.vertices, axis=0)}\n' + \
            f'max {np.max(self.vertices, axis=0)}\n' + \
            f'range {np.max(self.vertices, axis=0)-np.min(self.vertices, axis=0)}\n' + \
            f'num vertices {self.vertices.shape[0]}\n' + \
            f'num triangles {self.triangles.shape[0]}\n'



    def load(self, filename):
        self.mesh = o3d.io.read_triangle_mesh(filename)
        if not self.mesh.has_triangles():
            raise Exception('Triangle mesh expected.')
        self.__o3d_to_numpy()



    def demean(self):
        self.mesh.translate(-np.mean(np.asarray(self.mesh.vertices), axis=0))
        self.__o3d_to_numpy()



    def transform(self, T):
        self.mesh.rotate(T.GetRotationMatrix(), center=(0,0,0))
        self.mesh.translate(T.GetTranslation())
        self.__o3d_to_numpy()



    def __o3d_to_numpy(self):
        if not self.mesh.has_triangle_normals():
            self.mesh.compute_triangle_normals()
        if not self.mesh.has_vertex_normals():
            self.mesh.compute_vertex_normals()
        self.vertices = np.asarray(self.mesh.vertices)
        self.vertex_normals = np.asarray(self.mesh.vertex_normals)
        if self.mesh.has_vertex_colors():
            self.vertex_colors = np.asarray(self.mesh.vertex_colors)
        else:
            self.vertex_colors = None
        self.triangles = np.asarray(self.mesh.triangles)
        self.triangle_normals = np.asarray(self.mesh.triangle_normals)
        self.triangle_vertices = self.vertices[self.triangles]



    def __numpy_to_o3d(self):
        self.mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        self.mesh.vertex_normals = o3d.utility.Vector3dVector(self.vertex_normals)
        self.mesh.vertex_colors = o3d.utility.Vector3dVector(self.vertex_colors)
        self.mesh.triangles = o3d.utility.Vector3iVector(self.triangles)
        self.mesh.triangle_normals = o3d.utility.Vector3dVector(self.triangle_normals)



    def show(self, show_cs=True, show_vertices=True, show_normals=True):
        # Coordinate system
        if show_cs:
            cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=100.0, origin=[ 0.0, 0.0, 0.0 ])
        else:
            cs = None
        # Convert mesh to point cloud to visualize vertices and vertex normals
        if show_vertices:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self.vertices)
            if show_normals:
                pcd.normals = o3d.utility.Vector3dVector(self.vertex_normals)
        else:
            pcd = None
        # Visualize
        objects = [ self.mesh ]
        if cs is not None:
            objects.append(cs)
        if pcd is not None:
            objects.append(pcd)
        o3d.visualization.draw_geometries(objects, point_show_normal=True)



    def generateFromImageFile(self, imageFile, pixel_size=1.0):
        img = cv2.imread(imageFile, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.generateFromImage(img, pixel_size)



    def generateFromImage(self, img, pixel_size=1.0):
        self.vertices = np.zeros((4 * img.shape[0] * img.shape[1], 3))
        self.vertex_normals = np.zeros((4 * img.shape[0] * img.shape[1], 3))
        self.vertex_normals[:,2] = 1.0
        self.vertex_colors = np.zeros((4 * img.shape[0] * img.shape[1], 3))
        self.triangles = np.zeros((2 * img.shape[0] * img.shape[1], 3), dtype=int)
        self.triangle_normals = np.zeros((2 * img.shape[0] * img.shape[1], 3))
        self.triangle_normals[:,2] = 1.0
        print(img.shape)
        for r in range(img.shape[0]):
            for c in range(img.shape[1]):
                i = 4 * (r * img.shape[1] + c)
                self.vertices[i  ,0:2] = pixel_size * c,     pixel_size * r
                self.vertices[i+1,0:2] = pixel_size * c,     pixel_size * (r+1)
                self.vertices[i+2,0:2] = pixel_size * (c+1), pixel_size * r
                self.vertices[i+3,0:2] = pixel_size * (c+1), pixel_size * (r+1)
                self.vertex_colors[i:i+4,:] = img[img.shape[0]-r-1,c,:] / 255.0
                j = 2 * (r * img.shape[1] + c)
                self.triangles[j  ,:] = i+1, i, i+3
                self.triangles[j+1,:] = i+2, i+3, i
        self.triangle_vertices = self.vertices[self.triangles]
        self.__numpy_to_o3d()

