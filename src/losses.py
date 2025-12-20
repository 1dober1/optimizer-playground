import numpy as np


class MSE:
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X, w, y):
        y_pred = X @ w
        error = y_pred - y.flatten()
        return 2 / X.shape[0] * (X.T @ error)


class RMSE:
    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def gradient(self, X, w, y, eps=1e-12):
        y_pred = X @ w
        error = y_pred - y
        rmse = np.sqrt(np.mean(error**2)) + eps
        return (X.T @ error) / (X.shape[0] * rmse)


class MAE:
    def __call__(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def gradient(self, X, w, y):
        y_pred = X @ w
        error = np.abs(y_pred - y)
        return (X.T @ np.sign(error)) / X.shape[0]


class Huber:
    def __init__(self, delta=1.0):
        self.delta = delta

    def __call__(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta

        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (np.abs(error) - self.delta / 2)

        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def gradient(self, X, w, y):
        y_pred = X @ w
        error = y_pred - y
        is_small_error = np.abs(error) <= self.delta
        grad_error = np.where(
            is_small_error, error, self.delta * np.sign(error)
        )
        return (X.T @ grad_error) / X.shape[0]


class LogCosh:
    def __call__(self, y_true, y_pred):
        return np.mean(np.log(np.cosh(y_true - y_pred)))

    def gradient(self, X, w, y):
        y_pred = X @ w
        error = y_pred - y
        return (X.T @ np.tanh(error)) / X.shape[0]
