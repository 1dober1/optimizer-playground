import numpy as np


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")

        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v + self.lr * grad

        return w - self.v

    def reset(self):
        self.v = None


class NesterovMomentum:
    def __init__(self, lr=0.01, momentum=0.9):
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")

        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, w, grad):
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v + self.lr * grad

        return w - (self.momentum * self.v + self.lr * grad)

    def reset(self):
        self.v = None
