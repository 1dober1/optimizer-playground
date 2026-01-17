import numpy as np

from src.optimizers.adam import Adam
from src.optimizers.adamw import AdamW
from src.optimizers.momentum import Momentum, NesterovMomentum
from src.optimizers.rmsprop import RMSProp


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


def test_momentum_two_steps_constant_grad():
    w = np.array([0.0, 0.0])
    g = np.array([1.0, 1.0])

    lr = 0.1
    mu = 0.9
    opt = Momentum(lr=lr, momentum=mu)

    w1 = opt.step(w, g)
    assert np.allclose(w1, np.array([-0.1, -0.1]))

    w2 = opt.step(w1, g)
    assert np.allclose(w2, np.array([-0.29, -0.29]))


def test_nesterov_two_steps_constant_grad():
    w = np.array([0.0, 0.0])
    g = np.array([1.0, 1.0])

    lr = 0.1
    mu = 0.9
    opt = NesterovMomentum(lr=lr, momentum=mu)

    w1 = opt.step(w, g)
    assert np.allclose(w1, np.array([-0.19, -0.19]))

    w2 = opt.step(w1, g)
    assert np.allclose(w2, np.array([-0.461, -0.461]))


def test_momentum_reset():
    w = np.array([0.0, 0.0])
    g = np.array([1.0, 1.0])

    opt = Momentum(lr=0.1, momentum=0.9)
    _ = opt.step(w, g)
    assert opt.v is not None

    opt.reset()
    assert opt.v is None

    w1 = opt.step(w, g)
    assert np.allclose(w1, np.array([-0.1, -0.1]))


def test_rmsprop_two_steps_constant_grad():
    w = np.array([0.0, 0.0])
    g = np.array([1.0, 1.0])

    lr = 0.1
    beta = 0.9
    eps = 1e-8
    opt = RMSProp(lr=lr, beta=beta, epsilon=eps)

    w1 = opt.step(w, g)
    expected_w1 = np.array([-lr / np.sqrt(0.1), -lr / np.sqrt(0.1)])
    assert np.allclose(w1, expected_w1)

    w2 = opt.step(w1, g)
    expected_w2 = expected_w1 - np.array(
        [lr / np.sqrt(0.19), lr / np.sqrt(0.19)]
    )
    assert np.allclose(w2, expected_w2)


def test_rmsprop_reset():
    opt = RMSProp(lr=0.1)
    w = np.array([1.0, 2.0])
    g = np.array([0.5, -0.5])

    _ = opt.step(w, g)
    assert opt.s is not None

    opt.reset()
    assert opt.s is None
