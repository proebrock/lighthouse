import numpy as np



class RayTracerResult:
    """ Class to keep the results of a ray tracer session

    Contents:

    -> intersection_mask
    Mask with all rays intersecting with the mesh marked as True

    initial_points_cartesic
    3D points of first intersection with mesh;
    used in mirror ray tracer supporting reflections to keep
    track of the mirror surface; all other following results
    describe the final hit of the ray
    First intersections of the mesh are only tracked if the
    ray - after one or more reflections - hit a surface;
    in other words initial_points_cartesic and
    points_cartesic contain the same amount of points

    -> points_cartesic
    Final hit, intersection point in cartesic coordinates (x, y, z)

    -> points_barycentric
    Final hit, intersection point in homogenous barycentric coordiates
    (1.0 - u - v, u, v)

    -> triangle_indices
    Final hit, index of the triangle of the mesh hit by the ray

    -> scale
    Factor the ray direction has to be multiplied with in order to
    reach the intersection point; if the ray direction vector length
    of the ray tracing was normalized (length 1.0), scale contains the
    distance between the ray origin and the intersection point;
    for reflections, the total distance is summed up

    -> num_reflections
    Number of reflections the ray has made before hitting a solid surface
    """

    def __init__(self, num_rays=0):
        self.clear(num_rays)



    def __str__(self):
        return f'RayTracerResult({np.sum(self.intersection_mask)/self.intersection_mask.size} intersections)'



    def clear(self, num_rays=0):
        self.intersection_mask = np.zeros(num_rays, dtype=bool)
        self.initial_points_cartesic = np.zeros((num_rays, 3))
        self.points_cartesic = np.zeros((num_rays, 3))
        self.points_barycentric = np.zeros((num_rays, 3))
        self.triangle_indices = np.zeros(num_rays, dtype=int)
        self.scale = np.zeros(num_rays)
        self.num_reflections = np.zeros(num_rays, dtype=int)



    def reduce_to_valid(self):
        self.initial_points_cartesic = self.initial_points_cartesic[self.intersection_mask]
        self.points_cartesic = self.points_cartesic[self.intersection_mask]
        self.points_barycentric = self.points_barycentric[self.intersection_mask]
        self.triangle_indices = self.triangle_indices[self.intersection_mask]
        self.scale = self.scale[self.intersection_mask]
        self.num_reflections = self.num_reflections[self.intersection_mask]



    def expand(self):
        n = self.intersection_mask.size
        tmp = RayTracerResult(n)
        tmp.intersection_mask = self.intersection_mask
        tmp.initial_points_cartesic[self.intersection_mask] = self.initial_points_cartesic
        tmp.points_cartesic[self.intersection_mask] = self.points_cartesic
        tmp.points_barycentric[self.intersection_mask] = self.points_barycentric
        tmp.triangle_indices[self.intersection_mask] = self.triangle_indices
        tmp.scale[self.intersection_mask] = self.scale
        tmp.num_reflections[self.intersection_mask] = self.num_reflections
        self.__dict__.update(tmp.__dict__)
