import numpy as np
import open3d as o3d
from trafolib.trafo3d import Trafo3d



class SceneVisualizer:

    def __init__(self):
        self.geometry_list = []

    def add_coordinate_system(self, size=1.0, T=Trafo3d()):
        cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
        cs.rotate(T.get_rotation_matrix(), center=(0, 0, 0))
        cs.translate(T.get_translation())
        self.geometry_list.append(cs)

    def add_cam_cs(self, cam, size=1.0):
        T = cam.get_camera_position()
        self.add_coordinate_system(size, T)

    def add_cam_frustum(self, cam, size=1.0, color=(0, 0, 0)):
        shape = cam.get_chip_size()
        dimg = np.zeros((shape[1], shape[0]))
        dimg[:] = np.NaN
        dimg[0, 0] = size
        dimg[0, -1] = size
        dimg[-1, 0] = size
        dimg[-1, -1] = size
        P = cam.depth_image_to_scene_points(dimg)
        line_set = o3d.geometry.LineSet()
        points = np.vstack((cam.get_camera_position().get_translation(), P))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(color, (len(lines), 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.geometry_list.append(line_set)

    def add_cam_rays(self, cam, P, color=(0, 0, 0)):
        self.add_rays(cam.get_camera_position().get_translation(), P, color)

    def add_mesh(self, mesh):
        self.geometry_list.append(mesh.mesh)

    def add_points(self, P, color=(0, 0, 0)):
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(P)
        C = np.tile(color, (P.shape[0], 1))
        pcl.colors = o3d.utility.Vector3dVector(C)
        self.geometry_list.append(pcl)

    def add_rays(self, origin, P, color=(0, 0, 0)):
        line_set = o3d.geometry.LineSet()
        points = np.vstack((origin, P))
        line_set.points = o3d.utility.Vector3dVector(points)
        lines = np.zeros((points.shape[0], 2))
        lines[:, 1] = np.arange(1, points.shape[0]+1)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        colors = np.tile(color, (points.shape[0]+1, 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)
        self.geometry_list.append(line_set)

    def show(self):
        if len(self.geometry_list) == 0:
            return
        o3d.visualization.draw_geometries(self.geometry_list)
