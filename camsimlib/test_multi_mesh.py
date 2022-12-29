# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np



from . multi_mesh import MultiMesh



def test_mtindices_to_tindices():
    meshes = MultiMesh()
    meshes.triangle_mesh_indices = np.array((0, 3, 5))
    mt_indices = np.array((
        # (mesh index, triangle index)
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
    ))
    tindices = meshes.mtindices_to_tindices(mt_indices)
    assert np.all(tindices == (0, 1, 2, 3, 4, 5))



def test_tindices_to_mtindices():
    meshes = MultiMesh()
    meshes.triangle_mesh_indices = np.array((0, 3, 5))
    tindices = np.array((0, 1, 2, 3, 4, 5))
    mt_indices = meshes.tindices_to_mtindices(tindices)
    assert np.all(mt_indices == np.array((
        # (mesh index, triangle index)
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 0),
        (1, 1),
        (2, 0),
    )))
