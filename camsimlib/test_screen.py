import pytest
import numpy as np

from . screen import Screen



def test_screen_to_scene_external_image():
    """
                6pix, 30mm
             .----------------.
             | p[0]      p[2] |
             |                |
        5pix |     Image      |
        50mm |                |
             |                |
             | p[1]      p[3] |
             .----------------.
    """
    dimensions = (30.0, 50.0)
    image = np.zeros((5, 6, 3))
    screen = Screen(dimensions, image)
    indices = np.array(( # Screen coordinates
        (0.0, 0.0), # Top-left image point
        (4.0, 0.0), # Bottom-left image point
        (0.0, 5.0), # Top-right image point
        (4.0, 5.0), # Bottom-right image point
    ))
    P_expected = np.array((
        (2.5,  5.0, 0.0),
        (2.5,  45.0, 0.0),
        (27.5,  5.0, 0.0),
        (27.5, 45.0, 0.0),
    ))
    P = screen.image_indices_to_scene(indices)
    assert np.allclose(P, P_expected)
    on_chip_mask = screen.indices_on_chip_mask(indices)
    assert np.all(on_chip_mask)
