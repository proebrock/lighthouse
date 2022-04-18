# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
from re import T
import pytest
import numpy as np
from . ray_tracer_python import RayTracer as RayTracerPython
from . ray_tracer_embree import RayTracer as RayTracerEmbree

from . o3d_utils import mesh_generate_rays
import open3d as o3d # for visualization in debugging



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def visualize_scene(rayorigs, raydirs, vertices, triangles):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((1, 0, 0))
    rays = mesh_generate_rays(rayorigs, raydirs)
    o3d.visualization.draw_geometries([cs, mesh, rays])



def generate_rectangle(z=0):
    """ Generates a rectangle in the X/Y plane made from two triangles
    """
    vertices = np.array((
        ( 100,  100, z),
        (-100,  100, z),
        (-100, -100, z),
        ( 100, -100, z),
        ))
    triangles = np.array((
        (3, 0, 2),
        (1, 2, 0),
        ), dtype=int)
    return vertices, triangles



@pytest.fixture(params=[RayTracerPython, RayTracerEmbree])
def RayTracerImplementation(request):
    return request.param



def test_single_orig_single_dir(RayTracerImplementation):
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((10, 10, 10))
    raydirs = np.array((0, 0, -1))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([True], dtype=bool)
    assert np.allclose(rt.get_points_cartesic(), (10, 10, 0))
    assert np.allclose(rt.get_points_barycentric(), (0, 0.55, 0.45))
    assert np.allclose(rt.get_triangle_indices(), (0, ))
    assert np.allclose(rt.get_scale(), (10, ))



def test_single_orig_multi_dirs(RayTracerImplementation):
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((0, 0, 20))
    raydirs = np.array((
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, -1),
        ))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (20, 0, 0),
        (0, 20, 0),
        (-20, 0, 0),
        (0, -20, 0),
        )))
    assert np.allclose(rt.get_points_barycentric(), np.array((
        (0.1, 0.5, 0.4),
        (0.1, 0.4, 0.5),
        (0.1, 0.5, 0.4),
        (0.1, 0.4, 0.5),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (0, 1, 1, 0)
        )
    assert np.allclose(rt.get_scale(),
        (20, 20, 20, 20)
        )



def test_multi_origs_single_dir(RayTracerImplementation):
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((
        (10, 0, -5),
        (0, 10, -5),
        (-10, 0, -5),
        (0, -10, -5),
        ))
    raydirs = np.array((0, 0, 1))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (10, 0, 0),
        (0, 10, 0),
        (-10, 0, 0),
        (0, -10, 0),
        )))
    assert np.allclose(rt.get_points_barycentric(), np.array((
        (0.05, 0.5,  0.45),
        (0.05, 0.45, 0.5 ),
        (0.05, 0.5,  0.45),
        (0.05, 0.45, 0.5 ),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (0, 1, 1, 0)
        )
    assert np.allclose(rt.get_scale(),
        (5, 5, 5, 5)
        )



def test_multi_origs_multi_dirs(RayTracerImplementation):
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((
        (-10, 0, 10),
        (0, -10, 10),
        (10, 0, 10),
        (0, 10, 10),
        ))
    raydirs = np.array((
        (-1, 0, -1),
        (0, -2, -1),
        (3, 0, -1),
        (0, 4, -1),
        ))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (-20, 0, 0),
        (0, -30, 0),
        (40, 0, 0),
        (0, 50, 0),
        )))
    assert np.allclose(rt.get_points_barycentric(), np.array((
        (0.1,  0.5,  0.4),
        (0.15, 0.35, 0.5),
        (0.2,  0.5,  0.3),
        (0.25, 0.25, 0.5),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (1, 0, 0, 1)
        )
    assert np.allclose(rt.get_scale(),
        (10, 10, 10, 10)
        )



def test_invalid_origs_and_dirs(RayTracerImplementation):
    vertices, triangles = generate_rectangle()
    rayorigs = np.zeros((2, 3))
    raydirs = np.zeros((5, 3))
    with pytest.raises(ValueError):
        rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)



def test_no_intersect_empty_mesh(RayTracerImplementation):
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3), dtype=int)
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([False], dtype=bool)
    assert rt.get_points_cartesic().size == 0
    assert rt.get_points_barycentric().size == 0
    assert rt.get_triangle_indices().size == 0
    assert rt.get_scale().size == 0



def test_no_intersect_ray_misses(RayTracerImplementation):
    vertices, triangles = generate_rectangle(z=-10)
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([False], dtype=bool)
    assert rt.get_points_cartesic().size == 0
    assert rt.get_points_barycentric().size == 0
    assert rt.get_triangle_indices().size == 0
    assert rt.get_scale().size == 0



def test_shortest_intersection(RayTracerImplementation):
    btm_vertices, btm_triangles = generate_rectangle(z=-10.0)
    mid_vertices, mid_triangles = generate_rectangle(z=30.0)
    top_vertices, top_triangles = generate_rectangle(z=80.0)
    vertices = np.vstack((btm_vertices, mid_vertices, top_vertices))
    triangles = np.vstack((btm_triangles, mid_triangles+4, top_triangles+8))
    rayorigs = np.array((5, 5, 0))
    raydirs = np.array((0, 0, 3))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([True], dtype=bool)
    assert np.allclose(rt.get_points_cartesic(), (5, 5, 30))
    assert np.allclose(rt.get_scale(), (10,))



def test_raydir_length_and_scale(RayTracerImplementation):
    vertices, triangles = generate_rectangle(z=11.0)
    rayorigs = np.array((0, 0, 1))
    raydirs = np.array(((0, 0, 0.5), (0, 0, 1), (0, 0, 2)))
    rt = RayTracerImplementation(rayorigs, raydirs, vertices, triangles)
    rt.run()
    # We expect that the raytracer does not normalize the raydirs
    # that have been provided: rayorigs + raydirs * scale should
    # equal the intersection point
    np.allclose(rt.get_scale(), (20, 10, 5))



def generate_raydirs(num_lat, num_lon, lat_angle_deg):
    lat_angle = np.deg2rad(lat_angle_deg)
    lat_angle = np.clip(lat_angle, 0, np.pi)
    lat = np.linspace(0, lat_angle, num_lat)
    lon = np.linspace(0, 2 * np.pi, num_lon)
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    raydirs = np.zeros((num_lat, num_lon, 3))
    raydirs[:, :, 0] = np.sin(lat) * np.cos(lon)
    raydirs[:, :, 1] = np.sin(lat) * np.sin(lon)
    raydirs[:, :, 2] = np.cos(lat)
    return raydirs.reshape((-1, 3))



def test_two_implementations():
    # Setup scene: big sphere
    sphere_big = o3d.io.read_triangle_mesh('data/sphere.ply')
    if np.asarray(sphere_big.vertices).size == 0:
        raise Exception('Unable to load data file')
    sphere_big.compute_triangle_normals()
    sphere_big.compute_vertex_normals()
    sphere_big.scale(1, center=sphere_big.get_center())
    sphere_big.translate(-sphere_big.get_center())
    sphere_big.translate((0.5, -0.5, 5))
    # Setup scene: small sphere
    sphere_small = o3d.io.read_triangle_mesh('data/sphere.ply')
    if np.asarray(sphere_small.vertices).size == 0:
        raise Exception('Unable to load data file')
    sphere_small.compute_triangle_normals()
    sphere_small.compute_vertex_normals()
    sphere_small.scale(0.2, center=sphere_small.get_center())
    sphere_small.translate(-sphere_small.get_center())
    sphere_small.translate((-0.1, 0.1, 3))
    mesh = sphere_big + sphere_small
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    rayorigs = (0, 0, 0)
    raydirs = generate_raydirs(21, 41, 30)
    #visualize_scene(rayorigs, raydirs, vertices, triangles)
    # Run raytracers
    rt0 = RayTracerPython(rayorigs, raydirs, vertices, triangles)
    rt0.run()
    rt1 = RayTracerEmbree(rayorigs, raydirs, vertices, triangles)
    rt1.run()

    assert np.all( \
        rt0.get_intersection_mask() == \
        rt1.get_intersection_mask())
    assert np.allclose( \
        rt0.get_points_cartesic(), \
        rt1.get_points_cartesic())
    assert np.allclose( \
        rt0.get_points_barycentric(), \
        rt1.get_points_barycentric(), atol=1e-4)
    # This may not always be given: if due to rounding the raytracers
    # hit different neigboring triangles, the result may be still
    # visually correct!
    assert np.all( \
        rt0.get_triangle_indices() == \
        rt1.get_triangle_indices())
    assert np.allclose( \
        rt0.get_scale(), \
        rt1.get_scale())

    if False:
        # Verbose output
        coverage = (100 * np.sum(rt0.get_intersection_mask())) / \
            rt0.get_intersection_mask().size
        print(f'{coverage:.1f} percent of rays hit the object')
        print(f'distances in {np.min(rt0.get_scale()):.1f}..{np.max(rt0.get_scale()):.1f}')



if __name__ == '__main__':
    pytest.main()
