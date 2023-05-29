import numpy as np



def bundle_adjust(cams, points):
    assert len(cams) == points.shape[0]
    assert points.ndim == 3
    assert points.shape[2] == 2
    num_cams = len(cams)
    num_points = points.shape[1]


