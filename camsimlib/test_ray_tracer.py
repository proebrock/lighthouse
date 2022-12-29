# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import random as rand
import open3d as o3d
import numpy as np

from . multi_mesh import MultiMesh
from . rays import Rays
from . ray_tracer_result import RayTracerResult

from . ray_tracer_python import RayTracerPython
from . ray_tracer_embree import RayTracerEmbree
from . ray_tracer_mirrors import RayTracerMirrors

# for visualization in debugging
from . o3d_utils import mesh_generate_rays



# Reproducible tests with random numbers
rand.seed(0)
np.random.seed(0)



def visualize_scene(rayorigs, raydirs, meshlist):
    cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    rays = mesh_generate_rays(rayorigs, raydirs)
    object_list = [ cs, rays ]
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for i, mesh in enumerate(meshlist):
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh.paint_uniform_color(colors[i])
        object_list.append(mesh)
    o3d.visualization.draw_geometries(object_list)



def generate_rectangles(zs):
    """ Generates a rectangle in the X/Y plane made from two triangles
    """
    if not(isinstance(zs, list) or isinstance(zs, tuple)):
        zs = [ zs ]
    vertices_list = []
    triangles_list = []
    for z in zs:
        vertices_list.append(np.array((
            ( 100.0,  100.0, z),
            (-100.0,  100.0, z),
            (-100.0, -100.0, z),
            ( 100.0, -100.0, z),
            )))
        triangles_list.append(np.array((
            (3, 0, 2),
            (1, 2, 0),
            ), dtype=int))
    meshes = MultiMesh()
    meshes.from_components_lists(vertices_list, triangles_list)
    return meshes



@pytest.fixture(params=[RayTracerPython, RayTracerEmbree, RayTracerMirrors])
def RayTracerImplementation(request):
    return request.param



def test_single_orig_single_dir(RayTracerImplementation):
    mesh = generate_rectangles(0)
    rayorigs = np.array((10, 10, 10))
    raydirs = np.array((0, 0, -1))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert rt.r.intersection_mask == np.array([True], dtype=bool)
    assert np.allclose(rt.r.points_cartesic, (10, 10, 0))
    assert np.allclose(rt.r.points_barycentric, (0, 0.55, 0.45))
    assert np.allclose(rt.r.triangle_indices, (0, ))
    assert np.allclose(rt.r.scale, (10, ))
    assert np.allclose(rt.r.num_reflections, (0, ))




def test_single_orig_multi_dirs(RayTracerImplementation):
    mesh = generate_rectangles(0)
    rayorigs = np.array((0, 0, 20))
    raydirs = np.array((
        (1, 0, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (0, -1, -1),
        ))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert np.sum(rt.r.intersection_mask) == 4
    assert np.allclose(rt.r.points_cartesic, np.array((
        (20, 0, 0),
        (0, 20, 0),
        (-20, 0, 0),
        (0, -20, 0),
        )))
    assert np.allclose(rt.r.points_barycentric, np.array((
        (0.1, 0.5, 0.4),
        (0.1, 0.4, 0.5),
        (0.1, 0.5, 0.4),
        (0.1, 0.4, 0.5),
        )))
    assert np.allclose(rt.r.triangle_indices,
        (0, 1, 1, 0)
        )
    assert np.allclose(rt.r.scale,
        (20, 20, 20, 20)
        )
    assert np.allclose(rt.r.num_reflections,
        (0, 0, 0, 0)
        )



def test_multi_origs_single_dir(RayTracerImplementation):
    mesh = generate_rectangles(0)
    rayorigs = np.array((
        (10, 0, -5),
        (0, 10, -5),
        (-10, 0, -5),
        (0, -10, -5),
        ))
    raydirs = np.array((0, 0, 1))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert np.sum(rt.r.intersection_mask) == 4
    assert np.allclose(rt.r.points_cartesic, np.array((
        (10, 0, 0),
        (0, 10, 0),
        (-10, 0, 0),
        (0, -10, 0),
        )))
    assert np.allclose(rt.r.points_barycentric, np.array((
        (0.05, 0.5,  0.45),
        (0.05, 0.45, 0.5 ),
        (0.05, 0.5,  0.45),
        (0.05, 0.45, 0.5 ),
        )))
    assert np.allclose(rt.r.triangle_indices,
        (0, 1, 1, 0)
        )
    assert np.allclose(rt.r.scale,
        (5, 5, 5, 5)
        )
    assert np.allclose(rt.r.num_reflections,
        (0, 0, 0, 0)
        )



def test_multi_origs_multi_dirs(RayTracerImplementation):
    mesh = generate_rectangles(0)
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
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert np.sum(rt.r.intersection_mask) == 4
    assert np.allclose(rt.r.points_cartesic, np.array((
        (-20, 0, 0),
        (0, -30, 0),
        (40, 0, 0),
        (0, 50, 0),
        )))
    assert np.allclose(rt.r.points_barycentric, np.array((
        (0.1,  0.5,  0.4),
        (0.15, 0.35, 0.5),
        (0.2,  0.5,  0.3),
        (0.25, 0.25, 0.5),
        )))
    assert np.allclose(rt.r.triangle_indices,
        (1, 0, 0, 1)
        )
    assert np.allclose(rt.r.scale,
        (10, 10, 10, 10)
        )
    assert np.allclose(rt.r.num_reflections,
        (0, 0, 0, 0)
        )



def test_no_intersect_empty_meshlist(RayTracerImplementation):
    mesh = MultiMesh()
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert rt.r.intersection_mask == np.array([False], dtype=bool)
    assert rt.r.points_cartesic.size == 0
    assert rt.r.points_barycentric.size == 0
    assert rt.r.triangle_indices.size == 0
    assert rt.r.scale.size == 0
    assert rt.r.num_reflections.size == 0



def test_no_intersect_ray_misses(RayTracerImplementation):
    mesh = generate_rectangles(-10)
    rayorigs = np.array((0, 0, 0))
    raydirs = np.array((0, 0, 1))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    assert rt.r.intersection_mask == np.array([False], dtype=bool)
    assert rt.r.points_cartesic.size == 0
    assert rt.r.points_barycentric.size == 0
    assert rt.r.triangle_indices.size == 0
    assert rt.r.scale.size == 0
    assert rt.r.num_reflections.size == 0



def test_shortest_intersection(RayTracerImplementation):
    meshes = generate_rectangles((-10, 30, 80))
    rayorigs = np.array((5, 5, 0))
    raydirs = np.array((0, 0, 3))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, meshes)
    rt.run()
    assert rt.r.intersection_mask == np.array([True], dtype=bool)
    assert np.allclose(rt.r.points_cartesic, (5, 5, 30))
    assert np.allclose(rt.r.scale, (10,))



def test_raydir_length_and_scale(RayTracerImplementation):
    mesh = generate_rectangles(11.0)
    rayorigs = np.array((0, 0, 1))
    raydirs = np.array(((0, 0, 0.5), (0, 0, 1), (0, 0, 2)))
    rays = Rays(rayorigs, raydirs)
    rt = RayTracerImplementation(rays, mesh)
    rt.run()
    # We expect that the raytracer does not normalize the raydirs
    # that have been provided: rayorigs + raydirs * scale should
    # equal the intersection point
    np.allclose(rt.r.scale, (20, 10, 5))



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
    mesh_list = [ sphere_big, sphere_small ]
    meshes = MultiMesh(mesh_list)
    rayorigs = (0, 0, 0)
    raydirs = generate_raydirs(21, 41, 30)
    rays = Rays(rayorigs, raydirs)
    #visualize_scene(rayorigs, raydirs, meshlist)
    # Run raytracers
    rt0 = RayTracerPython(rays, meshes)
    rt0.run()
    rt1 = RayTracerEmbree(rays, meshes)
    rt1.run()

    assert np.all( \
        rt0.r.intersection_mask == \
        rt1.r.intersection_mask)
    assert np.allclose( \
        rt0.r.points_cartesic, \
        rt1.r.points_cartesic)
    assert np.allclose( \
        rt0.r.points_barycentric, \
        rt1.r.points_barycentric, atol=1e-4)
    # This may not always be given: if due to rounding the raytracers
    # hit different neigboring triangles, the result may be still
    # visually correct!
    assert np.all( \
        rt0.r.triangle_indices == \
        rt1.r.triangle_indices)
    assert np.allclose( \
        rt0.r.scale, \
        rt1.r.scale)



if __name__ == '__main__':
    pytest.main()
