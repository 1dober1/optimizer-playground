import numpy as np


class L2:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def __call__(self, w):
        return self.alpha * np.sum(w @ w)

    def grad(self, w):
        return 2 * self.alpha * w


class L1:
    def __init__(self, alpha):
        self.alpha = float(alpha)

    def __call__(self, w):
        return self.alpha * np.sum(np.abs(w))

    def prox(self, w, lr: float):
        t = lr * self.alpha
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


class Elastic_Net:
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        return self.alpha * self.l1_ratio * np.sum(
            np.abs(w)
        ) + 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(w @ w)

    def grad(self, w):
        return self.alpha * (1 - self.l1_ratio) * w

    def prox(self, w, lr: float):
        t = lr * self.alpha * self.l1_ratio
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
