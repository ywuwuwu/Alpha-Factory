import numpy as np
from alphafactory.allocator.online_alm import project_to_l1_ball


def test_project_to_l1_ball_basic():
    v = np.array([1.0, 2.0, -1.0])
    w = project_to_l1_ball(v, z=1.0)
    assert np.all(w >= 0)
    assert w.sum() <= 1.0 + 1e-8


def test_project_to_l1_ball_no_change():
    v = np.array([0.2, 0.3, 0.1])
    w = project_to_l1_ball(v, z=1.0)
    assert np.allclose(v, w)
