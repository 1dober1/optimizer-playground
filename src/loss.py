import numpy as np


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def grad_mean_squared_error(X, w, y):
    return 2 / X.shape[0] * (X.T @ X @ w - X.T @ y)
