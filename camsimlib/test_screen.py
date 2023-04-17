import pytest
import numpy as np

from . screen import Screen



def test_screen_to_scene_external_image():
    """
                6pix, 60mm
             .----------------.
             | p[0]      p[2] |
             |                |
        5pix |     Image      |
        50mm |                |
             |                |
             | p[1]      p[3] |
             .----------------.
    """
    image = np.zeros((5, 6, 3))
    width = 60.0
    height = 50.0
    screen = Screen((width, height), image)
    p = np.array(( # Screen coordinates
        (0.0, 0.0), # Top-left image point
        (4.0, 0.0), # Bottom-left image point
        (0.0, 5.0), # Top-right image point
        (4.0, 5.0), # Bottom-right image point
    ))
    P_expected = np.array((
        (5.0, 5.0, 0.0),
        (5.0, 45.0, 0.0),
        (55.0, 5.0, 0.0),
        (55.0, 45.0, 0.0),
    ))
    P = screen.screen_to_scene(p)
    assert np.allclose(P, P_expected)



def test_screen_to_scene_internal_image():
    """
                6pix, 60mm
             .----------------.
             | p[0]      p[2] |
             |                |
        5pix |     Image      |
        50mm |                |
             |                |
             | p[1]      p[3] |
             .----------------.
    """
    width = 60.0
    height = 50.0
    screen = Screen((width, height), (5, 6))
    p = np.array(( # Screen coordinates
        (0.0, 0.0), # Top-left image point
        (4.0, 0.0), # Bottom-left image point
        (0.0, 5.0), # Top-right image point
        (4.0, 5.0), # Bottom-right image point
    ))
    P_expected = np.array((
        (5.0, 5.0, 0.0),
        (5.0, 45.0, 0.0),
        (55.0, 5.0, 0.0),
        (55.0, 45.0, 0.0),
    ))
    P = screen.screen_to_scene(p)
    assert np.allclose(P, P_expected)



def test_invalid_screen_points():
    """
                6pix, 60mm
             .----------------.
             | p[0]      p[2] |
             |                |
        5pix |     Image      |
        50mm |                |
             |                |
             | p[1]      p[3] |
             .----------------.
    """
    image = np.zeros((5, 6, 3))
    width = 60.0
    height = 50.0
    screen = Screen((width, height), image)
    p = np.array(( # Screen coordinates
        (-0.5, -0.5), # Top-left image point
        (image.shape[0]-0.5, -0.5), # Bottom-left image point
        (-0.5, image.shape[1]-0.5), # Top-right image point
        (image.shape[0]-0.5, image.shape[1]-0.5), # Bottom-right image point
    ))
    P_expected = np.array((
        (0.0, 0.0, 0.0),
        (0.0, height, 0.0),
        (width, 0.0, 0.0),
        (width, height, 0.0),
    ))
    # This setup should still valid
    P = screen.screen_to_scene(p)
    assert np.allclose(P, P_expected)
    # This should be not
    with pytest.raises(Exception):
        _p = p.copy()
        _p[0, 0] = image.shape[0]
        P = screen.screen_to_scene(_p)
    # This should be not
    with pytest.raises(Exception):
        _p = p.copy()
        _p[1, 1] = image.shape[1]
        P = screen.screen_to_scene(_p)
