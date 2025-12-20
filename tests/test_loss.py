import numpy as np
from src.losses import MAE, MSE, Huber, LogCosh


def numerical_grad(loss_fn, X, w, y, eps=1e-6):
    grad = np.zeros_like(w)
    for i in range(len(w)):
        w1 = w.copy()
        w1[i] += eps
        w2 = w.copy()
        w2[i] -= eps
        f1 = loss_fn(y, X @ w1)
        f2 = loss_fn(y, X @ w2)
        grad[i] = (f1 - f2) / (2 * eps)
    return grad


def rel_error(a, b):
    return np.linalg.norm(a - b) / (
        np.linalg.norm(a) + np.linalg.norm(b) + 1e-12
    )


def test_mse_gradient_matches_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.normal(size=5)

    loss = MSE()
    g_a = loss.gradient(X, w, y)
    g_n = numerical_grad(loss, X, w, y)

    assert rel_error(g_a, g_n) < 1e-4


def test_mae_gradient_matches_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.normal(size=5)

    loss = MAE()
    g_a = loss.gradient(X, w, y)
    g_n = numerical_grad(loss, X, w, y, eps=1e-5)

    assert rel_error(g_a, g_n) < 1e-3


def test_huber_gradient_matches_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.normal(size=5)

    loss = Huber()
    g_a = loss.gradient(X, w, y)
    g_n = numerical_grad(loss, X, w, y)

    assert rel_error(g_a, g_n) < 1e-4


def test_logcosh_gradient_matches_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.normal(size=5)

    loss = LogCosh()
    g_a = loss.gradient(X, w, y)
    g_n = numerical_grad(loss, X, w, y)

    assert rel_error(g_a, g_n) < 1e-4
