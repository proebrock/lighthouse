# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import random as rand
import open3d as o3d
import numpy as np

from . ray_tracer_mirrors import RayTracerMirrors

# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def generate_single_triangle(vertices):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    triangles = np.array([[0, 1, 2]], dtype=int)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_triangle_normals()
    return mesh



def test_basic_setup():
    # Generate mesh list
    a = 100.0
    # Along x axis facing north
    t0 = generate_single_triangle(
        [[0, 0, 0], [a/2, 0, a/2], [a, 0, 0]])
    assert np.all(np.asarray(t0.triangle_normals) == (0, 1, 0))
    # Along y axis facing east
    t1 = generate_single_triangle(
        [[0, 0, 0], [0, a, 0], [0, a/2, a/2]])
    assert np.all(np.asarray(t1.triangle_normals) == (1, 0, 0))
    # Parallel to x axis facing south
    t2 = generate_single_triangle(
        [[0, a, 0], [a, a, 0], [a/2, a, a/2]])
    assert np.all(np.asarray(t2.triangle_normals) == (0, -1, 0))
    mesh_list = [ t0, t1, t2 ]
    mirrors = [ False, True, True ]
    # Generate rays
    rayorigs = np.array([
        [a, a/2, 1.0],
        [a, a/2, 1.0],
        [a, a/2, 1.0],
        [a, a/2, 1.0],
        [a/2, 1, 1.0],
        [a, a/2, 1.0],
        ])
    raydirs = np.array([
        [  1,   0, 0.0 ], # miss
        [ -1,  -1, 0.0 ], # hit
        [ -1,   0, 0.0 ], # mirror, miss
        [ -0.1, 1, 0.0 ], # mirror, hit
        [ -1,   1, 0.0 ], # mirror, mirror, miss
        [ -1,   1, 0.0 ], # mirror, mirror, hit
        ])
    # Normalize raydirs
    raydirslen = np.sqrt(np.sum(np.square(raydirs), axis=1))
    raydirs /= raydirslen[:, np.newaxis]
    # Run raytracing
    rt = RayTracerMirrors(rayorigs, raydirs, mesh_list, mirrors)
    rt.run()
    # Check results
    assert np.all(rt.get_intersection_mask() ==
        [ False, True, False, True, False, True])
    assert np.allclose(rt.get_points_cartesic(), np.array([
        [ a/2, 0, 1.0 ],
        [ 85, 0, 1.0 ],
        [ a/2, 0, 1.0 ],
        ]), atol=1e-5)
    assert np.all(rt.get_mesh_indices() ==
        [ 0, 0, 0 ])
    assert np.all(rt.get_triangle_indices() ==
        [ 0, 0, 0 ])
    assert np.allclose(rt.get_scale(), np.array([
        a/np.sqrt(2), 150.74813461, 3*a/np.sqrt(2)
        ]))
    assert np.all(rt.get_num_reflections() ==
        [ 0, 1, 2 ])



def test_mirror_front_and_backside():
    # Generate mesh list
    a = 100.0
    # Three parallel triangles, 1st: Below x axis facing north
    t0 = generate_single_triangle(
        [[0, -a/2, 0], [a/2, -a/2, a/2], [a, -a/2, 0]])
    assert np.all(np.asarray(t0.triangle_normals) == (0, 1, 0))
    # Three parallel triangles, 2nd: Along x axis facing north
    t1 = generate_single_triangle(
        [[0, 0, 0], [a/2, 0, a/2], [a, 0, 0]])
    assert np.all(np.asarray(t1.triangle_normals) == (0, 1, 0))
    # Three parallel triangles, 3rd: Above x axis facing south
    t2 = generate_single_triangle(
        [[0, a/2, 0], [a, a/2, 0], [a/2, a/2, a/2]])
    assert np.all(np.asarray(t2.triangle_normals) == (0, -1, 0))
    mesh_list = [ t0, t1, t2 ]
    mirrors = [ False, True, False ]
    # Generate rays
    rayorigs = np.array([
        [a/2, a/4, 1.0],
        [a/2, -a/4, 1.0],
        ])
    raydirs = np.array([
        [  0, -1, 0.0 ], # hitting mirror from above, angle normal<->ray is 180 deg
        [  0,  1, 0.0 ], # hitting mirror from below, angle normal<->ray is 0 deg
        ])
    # Normalize raydirs
    raydirslen = np.sqrt(np.sum(np.square(raydirs), axis=1))
    raydirs /= raydirslen[:, np.newaxis]
    # Run raytracing
    rt = RayTracerMirrors(rayorigs, raydirs, mesh_list, mirrors)
    rt.run()
    # Check results: Default behavior: mirror meshes reflect on both sides
    assert np.all(rt.get_intersection_mask() ==
        [ True, True ])
    assert np.allclose(rt.get_points_cartesic(), np.array([
        [ a/2,  a/2, 1.0 ],
        [ a/2, -a/2, 1.0 ],
        ]), atol=1e-5)
    assert np.all(rt.get_mesh_indices() ==
        [ 2, 0 ])
    assert np.all(rt.get_triangle_indices() ==
        [ 0, 0 ])
    assert np.allclose(rt.get_scale(), np.array([
        3*a/4, 3*a/4
        ]))
    assert np.all(rt.get_num_reflections() ==
        [ 1, 1 ])



def test_infinite_ray():
    # Generate mesh list
    a = 100.0
    # Along x axis facing north
    t0 = generate_single_triangle(
        [[0, 0, 0], [a/2, 0, a/2], [a, 0, 0]])
    assert np.all(np.asarray(t0.triangle_normals) == (0, 1, 0))
    # Along y axis facing east
    t1 = generate_single_triangle(
        [[0, 0, 0], [0, a, 0], [0, a/2, a/2]])
    assert np.all(np.asarray(t1.triangle_normals) == (1, 0, 0))
    # Parallel to x axis facing south
    t2 = generate_single_triangle(
        [[0, a, 0], [a, a, 0], [a/2, a, a/2]])
    assert np.all(np.asarray(t2.triangle_normals) == (0, -1, 0))
    mesh_list = [ t0, t1, t2 ]
    mirrors = [ True, False, True ]
    # Generate rays
    rayorigs = np.array([
        [ a/2, a/2, 1.0],
        [ a/2, a/2, 1.0],
        [ a/2, a/2, 1.0],
        ])
    raydirs = np.array([
        [ -1, 0, 0.0 ], # hit
        [  1, 0, 0.0 ], # is mirrored infinitely
        [  0, 1, 0.0 ], # miss
        ])
    # Normalize raydirs
    raydirslen = np.sqrt(np.sum(np.square(raydirs), axis=1))
    raydirs /= raydirslen[:, np.newaxis]
    # Run raytracing
    rt = RayTracerMirrors(rayorigs, raydirs, mesh_list, mirrors)
    rt.run()
    # Check results: Infinitely reflected ray should be discarded
    assert np.all(rt.get_intersection_mask() ==
        [ True, False, False ])
    assert np.allclose(rt.get_points_cartesic(), np.array([
        [ 0,  a/2, 1.0 ],
        ]))
    assert np.all(rt.get_mesh_indices() ==
        [ 1 ])
    assert np.all(rt.get_triangle_indices() ==
        [ 0 ])
    assert np.allclose(rt.get_scale(), np.array([
        a/2
        ]))
    assert np.all(rt.get_num_reflections() ==
        [ 0 ])
