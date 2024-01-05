import pytest
import numpy as np

from common.pose_graph import PoseGraph
from trafolib.trafo3d import Trafo3d



def test_load_save():
    g0 = PoseGraph()
    g0.add_vertex('v0')
    g0.add_vertex('v1')
    g0.add_edge('v0', 'v1', Trafo3d(t=(1,-2,3), rpy=(np.pi, 0, 0.1)))
    g0.add_edge('v0', 'v1', Trafo3d(t=(1,-2,2.5), rpy=(-np.pi, 0, 0.11)))
    param_dict0 = {}
    g0.dict_save(param_dict0)
    g1 = PoseGraph()
    g1.dict_load(param_dict0)
    param_dict1 = {}
    g1.dict_save(param_dict1)
    assert param_dict0 == param_dict1



def test_is_connected():
    g = PoseGraph()
    assert not g.is_connected()
    g.add_vertex('a')
    assert g.is_connected()
    g.add_vertex('b')
    assert not g.is_connected()
    g.add_edge('a', 'b')
    assert g.is_connected()



def test_add_duplicate_vertex():
    g = PoseGraph()
    g.add_vertex('v0')
    g.add_vertex('v1')
    with pytest.raises(ValueError):
        g.add_vertex('v1')



def test_add_edge_to_nonexistent_vertex():
    g = PoseGraph()
    g.add_vertex('v0')
    g.add_vertex('v1')
    g.add_edge('v0', 'v1')
    with pytest.raises(ValueError):
        g.add_edge('v0', 'v2')



def test_calculate_trafo_between_vertices_no_path_exists():
    g = PoseGraph()
    g.add_vertex('a')
    g.add_vertex('b')
    with pytest.raises(Exception):
        g.calculate_trafo_between_vertices('a', 'b')



def test_calculate_trafo_between_vertices_vertex_not_exist():
    g = PoseGraph()
    g.add_vertex('a')
    g.add_vertex('b')
    with pytest.raises(Exception):
        g.calculate_trafo_between_vertices('a', 'c')



def test_calculate_trafo_between_vertices():
    a_to_b = Trafo3d(t=(1,1.223,-56), rpy=(0.1, 2.2, -1.5))
    b_to_c = Trafo3d(t=(13,-11,0), rpy=(3.1, -1.2, -1.1))
    a_to_c = a_to_b*b_to_c
    g = PoseGraph()
    g.add_vertex('a')
    g.add_vertex('b')
    g.add_vertex('c')
    g.add_edge('a', 'b', a_to_b)
    g.add_edge('b', 'c', b_to_c)
    g_a_to_c = g.calculate_trafo_between_vertices('a', 'c')
    dt, dr = g_a_to_c.distance(a_to_c)
    assert np.isclose(dt, 0)
    assert np.isclose(dr, 0)



def test_calculate_trafo_between_vertices_inverse():
    a_to_b = Trafo3d(t=(1,1.223,-56), rpy=(0.1, 2.2, -1.5))
    g = PoseGraph()
    g.add_vertex('a')
    g.add_vertex('b')
    g.add_edge('a', 'b', a_to_b)
    g_b_to_a = g.calculate_trafo_between_vertices('b', 'a')
    dt, dr = g_b_to_a.distance(a_to_b.inverse())
    assert np.isclose(dt, 0)
    assert np.isclose(dr, 0)



def test_calculate_trafos_from_source_to_targets():
    a_to_b = Trafo3d(t=(1,1.223,-56), rpy=(0.1, 2.2, -1.5))
    b_to_c = Trafo3d(t=(13,-11,0), rpy=(3.1, -1.2, -1.1))
    a_to_d = Trafo3d(t=(3,-1,22), rpy=(1.1, 0.2, -1.7))
    g = PoseGraph()
    g.add_vertices(['a', 'b', 'c', 'd'])
    g.add_edge('a', 'b', a_to_b)
    g.add_edge('b', 'c', b_to_c)
    g.add_edge('a', 'd', a_to_d)
    trafos = g.calculate_trafos_from_source_to_targets('b', ['a', 'b', 'c', 'd'])
    expected_trafos = [
        a_to_b.inverse(),
        Trafo3d(),
        b_to_c,
        a_to_b.inverse()*a_to_d
    ]
    for t, et in zip(trafos, expected_trafos):
        dt, dr = et.distance(t)
        assert np.isclose(dt, 0)
        assert np.isclose(dr, 0)
