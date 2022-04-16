# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
import pytest
import numpy as np
from . ray_tracer_embree import RayTracer
#import open3d as o3d # for visualization in debugging



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def visualize_scene(rayorigs, raydirs, vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])



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



def test_single_orig_single_dir():
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((10, 10, 10))
    raydirs = np.array((0, 0, -1))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([True], dtype=bool)
    assert np.allclose(rt.get_points_cartesic(), (10, 10, 0))
    assert np.allclose(rt.get_triangle_indices(), (0,))



def test_single_orig_multi_dirs():
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((0, 0, 20))
    raydirs = np.array((
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, -1),
        ))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (20, 0, 0),
        (0, 20, 0),
        (-20, 0, 0),
        (0, -20, 0),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (0, 1, 1, 0)
        )



def test_multi_origs_single_dir():
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((
        (10, 0, -5),
        (0, 10, -5),
        (-10, 0, -5),
        (0, -10, -5),
        ))
    raydirs = np.array((0, 0, 1))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (10, 0, 0),
        (0, 10, 0),
        (-10, 0, 0),
        (0, -10, 0),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (0, 1, 1, 0)
        )



def test_multi_origs_multi_dirs():
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
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert np.sum(rt.get_intersection_mask()) == 4
    assert np.allclose(rt.get_points_cartesic(), np.array((
        (-20, 0, 0),
        (0, -30, 0),
        (40, 0, 0),
        (0, 50, 0),
        )))
    assert np.allclose(rt.get_triangle_indices(),
        (1, 0, 0, 1)
        )



def test_invalid_origs_and_dirs():
    vertices, triangles = generate_rectangle()
    rayorigs = np.zeros((2, 3))
    raydirs = np.zeros((5, 3))
    with pytest.raises(ValueError):
        rt = RayTracer(rayorigs, raydirs, vertices, triangles)



@pytest.mark.skip(reason="embree version fails")
def test_no_intersect_empty_mesh():
    vertices = np.zeros((0, 3))
    triangles = np.zeros((0, 3), dtype=int)
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([False], dtype=bool)
    assert rt.get_points_cartesic().size == 0
    assert rt.get_points_barycentric().size == 0
    assert rt.get_triangle_indices().size == 0
    assert rt.get_scale().size == 0



def test_no_intersect_ray_misses():
    vertices, triangles = generate_rectangle(z=-10)
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([False], dtype=bool)
    assert rt.get_points_cartesic().size == 0
    assert rt.get_points_barycentric().size == 0
    assert rt.get_triangle_indices().size == 0
    assert rt.get_scale().size == 0



def test_shortest_intersection():
    btm_vertices, btm_triangles = generate_rectangle(z=-10.0)
    mid_vertices, mid_triangles = generate_rectangle(z=30.0)
    top_vertices, top_triangles = generate_rectangle(z=80.0)
    vertices = np.vstack((btm_vertices, mid_vertices, top_vertices))
    triangles = np.vstack((btm_triangles, mid_triangles+4, top_triangles+8))
    rayorigs = np.array((5, 5, 0))
    raydirs = np.array((0, 0, 3))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
    assert rt.get_intersection_mask() == np.array([True], dtype=bool)
    assert np.allclose(rt.get_points_cartesic(), (5, 5, 30))
    assert np.allclose(rt.get_scale(), (10,))
