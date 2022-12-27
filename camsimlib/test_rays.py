# Start in Ubuntu similar to: py.test-3 -s --verbose
import pytest
import numpy as np

from . rays import Rays



def test_invalid_origs_and_dirs():
    rayorigs = np.zeros((2, 3))
    raydirs = np.zeros((5, 3))
    with pytest.raises(ValueError):
        rays = Rays(rayorigs, raydirs)
