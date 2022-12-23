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
    # Generate mesh lists
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
    rayorigs = np.array([[a, a/2, 1.0]])
    raydirs = np.array([
        [1, 0, 0], # miss
        [-1, -1 , 0], # hit
        [-1, 0, 0], # mirror, miss
        [-1, 1, 0], # mirror, mirror, hit
    ], dtype=np.float64)
    # Normalize raydirs
    raydirslen = np.sqrt(np.sum(np.square(raydirs), axis=1))
    raydirs /= raydirslen[:, np.newaxis]
    # Run raytracing
    rt = RayTracerMirrors(rayorigs, raydirs, mesh_list, mirrors)
    rt.run()



def test_mirror_front_and_backside():
    # cases:
    # 1) front side: mirror
    # 2) back side: ray hits mirror, we check thats its the backside, we
    #    initiate another raytrace starting at the front of the mirror
    # 3) Or: Backside of mirror is "absorbing ray" and we stop raytracing
    #    there
    pass



def test_inifinite_ray():
    # two mirroring triangles facing each other
    # ray passes through backside of first and bounces back and forth
    pass
