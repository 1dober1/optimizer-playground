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

    def grad(self, w):
        return self.alpha * np.sign(w)


class Elastic_Net:
    def __init__(self, alpha, l1_ratio):
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        return self.alpha * self.l1_ratio * np.sum(np.abs(w)) + self.alpha * (
            (1 - self.l1_ratio) / 2
        ) * np.sum(w @ w)

    def grad(self, w):
        return (
            self.alpha * self.l1_ratio * np.sign(w)
            + self.alpha * ((1 - self.l1_ratio) / 2) * 2 * w
        )
