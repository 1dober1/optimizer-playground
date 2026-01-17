import numpy as np

from src.optimizers.adam import Adam
from src.optimizers.adamw import AdamW


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


def test_adamw_reset_clears_state():
    opt = AdamW(lr=1e-2, weight_decay=0.1)
    w = np.array([1.0, 2.0])
    g = np.array([0.1, -0.2])

    opt.step(w, g)
    assert getattr(opt, "t", 0) != 0

    opt.reset()
    assert getattr(opt, "t", 0) == 0
    assert opt.m is None
    assert opt.v is None


def test_adamw_equals_adam_when_weight_decay_zero():
    w = np.array([1.0, 2.0, -3.0])
    g = np.array([0.3, -0.1, 0.2])

    adam = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    adamw = AdamW(
        lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-8, weight_decay=0.0
    )

    w_adam = adam.step(w, g)
    w_adamw = adamw.step(w, g)

    assert np.allclose(w_adam, w_adamw)


def test_adamw_weight_decay_shrinks_weights_but_not_intercept_vector():
    w = np.array([1.0, 2.0, 3.0])
    g = np.zeros_like(w)

    lr = 0.1
    wd = 0.01
    opt = AdamW(lr=lr, weight_decay=wd, exclude_intercept=True)

    w_next = opt.step(w, g)

    expected = w.copy()
    expected[0] = 1.0
    expected[1] = 2.0 - lr * wd * 2.0
    expected[2] = 3.0 - lr * wd * 3.0

    assert np.allclose(w_next, expected)


def test_adamw_weight_decay_excludes_intercept_matrix():
    w = np.array(
        [
            [1.0, -1.0],
            [2.0, -2.0],
            [3.0, -3.0],
        ]
    )
    g = np.zeros_like(w)

    lr = 0.1
    wd = 0.01
    opt = AdamW(lr=lr, weight_decay=wd, exclude_intercept=True)

    w_next = opt.step(w, g)

    expected = w.copy()
    expected[1:, :] = w[1:, :] - lr * wd * w[1:, :]

    assert np.allclose(w_next, expected)
