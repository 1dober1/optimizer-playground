import numpy as np
from src.loss import MSE


def numerical_grad(X, w, y, eps=1e-6):
    grad = np.zeros_like(w)
    loss = MSE()
    for i in range(len(w)):
        w1 = w.copy()
        w1[i] += eps
        w2 = w.copy()
        w2[i] -= eps
        f1 = loss(y, X @ w1)
        f2 = loss(y, X @ w2)
        grad[i] = (f1 - f2) / (2 * eps)
    return grad


def test_mse_gradient_matches_finite_diff():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = rng.normal(size=5)

    loss = MSE()
    g_analytical = loss.gradient(X, w, y)
    g_numeric = numerical_grad(X, w, y)

    rel_err = np.linalg.norm(g_analytical - g_numeric) / (
        np.linalg.norm(g_analytical) + np.linalg.norm(g_numeric) + 1e-12
    )
    assert rel_err < 1e-4
