import numpy as np


class Adam:
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad**2)

        m_hat = self.m / (1 - self.beta_1**self.t)
        v_hat = self.v / (1 - self.beta_2**self.t)

        return w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
