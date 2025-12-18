import numpy as np


class MSE:
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X, w, y):
        y_pred = X @ w
        error = y_pred - y.flatten()
        return 2 / X.shape[0] * (X.T @ error)
