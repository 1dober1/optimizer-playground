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
