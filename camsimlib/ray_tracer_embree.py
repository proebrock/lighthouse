import numpy as np
import open3d as o3d



class RayTracer:

    def __init__(self, rayorigs, raydirs, vertices, triangles):
        """ Intersection of multiple rays with a number of triangles
        :param rayorigs: Ray origins, shape (3,) or (3, n) for n rays
        :param raydir: Ray directions, shape (3,) or (3, n), for n rays
        :param vertices: Vertices, shape (k, 3)
        :param triangles: Triangle indices, shape (l, 3)
        """
        # Ray tracer results
        self.intersection_mask = None
        self.points_cartesic = None
        self.points_barycentric = None
        self.triangle_indices = None
        self.scale = None



    def get_intersection_mask(self):
        """ Get intersection mask: True for all rays that do intersect
        :return: Intersection mask of shape (m, ), type bool
        """
        return self.intersection_mask



    def get_points_cartesic(self):
        """ Get intersection points of rays with triangle in Cartesian coordinates (x, y, z)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self.points_cartesic



    def get_points_barycentric(self):
        """ Get intersection points of rays within triangle in barycentric coordinates (1-u-v, u, v)
        :return: Points of shape (k,3), k number of intersecting rays, k<=m
        """
        return self.points_barycentric



    def get_triangle_indices(self):
        """ Get indices of triangles intersecting with rays (0..n-1) or -1
        :return: Indices of shape (k,), type int, k number of intersecting rays, k<=m
        """
        return self.triangle_indices



    def get_scale(self):
        """ Get scale so that "self.rayorigs + scale * self.raydir" equals the intersection point
        :return: Scale of shape (k,), k number of intersecting rays, k<=m
        """
        return self.scale





#cube = o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh.create_box())
#scene = o3d.t.geometry.RaycastingScene()
#scene.add_triangles(cube)

# Use a helper function to create rays for a pinhole camera.
#rays = scene.create_rays_pinhole(fov_deg=60, center=[0.5,0.5,0.5], eye=[-1,-1,-1], up=[0,0,1],
#                               width_px=320, height_px=240)

# Compute the ray intersections and visualize the hit distance (depth)
#ans = scene.cast_rays(rays)
#plt.imshow(ans['t_hit'].numpy())
