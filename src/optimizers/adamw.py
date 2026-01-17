import numpy as np


class AdamW:
    decoupled_weight_decay = True

    def __init__(
        self,
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
        exclude_intercept=True,
    ):
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if beta_1 <= 0 or beta_1 >= 1:
            raise ValueError("beta_1 must be in (0, 1)")
        if beta_2 <= 0 or beta_2 >= 1:
            raise ValueError("beta_2 must be in (0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")
        if weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.exclude_intercept = exclude_intercept

        self.m = None
        self.v = None
        self.t = 0

    def _decay_term(self, w):
        if not self.exclude_intercept:
            return w

        decay_w = w.copy()
        if decay_w.ndim == 1:
            decay_w[0] = 0.0
        elif decay_w.ndim == 2:
            decay_w[0, :] = 0.0
        else:
            decay_w[0] = 0.0
        return decay_w

    def step(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grad
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * (grad**2)

        m_hat = self.m / (1 - self.beta_1**self.t)
        v_hat = self.v / (1 - self.beta_2**self.t)

        w_next = w - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

        if self.weight_decay != 0.0:
            w_next = w_next - self.lr * self.weight_decay * self._decay_term(w)

        return w_next

    def reset(self):
        self.m = None
        self.v = None
        self.t = 0
