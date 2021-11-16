# -*- coding: utf-8 -*-


# Start in Ubuntu similar to: py.test-3 -s --verbose
import random as rand
import pytest
import numpy as np
from . ray_tracer import RayTracer



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def generate_rectangle():
    """ Generates a rectangle in the X/Y plane made from two triangles
    """
    vertices = np.array((
        ( 100,  100, 0),
        (-100,  100, 0),
        (-100, -100, 0),
        ( 100, -100, 0),
        ))
    triangles = np.array((
        (3, 0, 2),
        (1, 2, 0),
        ))
    return vertices, triangles



def test_single_orig_single_dir():
    vertices, triangles = generate_rectangle()
    rayorigs = np.array((10, 10, 10))
    raydirs = np.array((0, 0, -1))
    rt = RayTracer(rayorigs, raydirs, vertices, triangles)
    rt.run()
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
