import numpy as np
from src.optimizers.adam import Adam


def test_adam_reset_clears_state():
    opt = Adam(lr=1e-2)
    w = np.array([1.0, 2.0])
    g = np.array([0.1, -0.2])

    opt.step(w, g)
    assert getattr(opt, "t", 0) != 0

    opt.reset()
    assert getattr(opt, "t", 0) == 0
    assert opt.m is None
    assert opt.v is None
