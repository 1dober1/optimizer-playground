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
        error = y_pred - y
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


class LogLoss:
    def __call__(self, y_true, y_logits):
        return np.mean(np.logaddexp(0, y_logits) - y_true * y_logits)

    def gradient(self, X, w, y_true):
        z = X @ w

        p = np.empty_like(z, dtype=float)
        pos = z >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        p[~pos] = ez / (1.0 + ez)

        error = p - y_true
        return (X.T @ error) / X.shape[0]


class CrossEntropyLoss:
    def __call__(self, y_true, y_logits):
        logits_max = np.max(y_logits, axis=1, keepdims=True)
        logits_shifted = y_logits - logits_max

        exp_logits = np.exp(logits_shifted)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_probs = logits_shifted - log_sum_exp

        return -np.mean(np.sum(y_true * log_probs, axis=1))

    def gradient(self, X, w, y_true):
        logits = X @ w
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        error = probs - y_true

        return (X.T @ error) / X.shape[0]
