import numpy as np


class Adam:
    """
    Adam optimizer.

    Adaptive Moment Estimation (Adam) computes adaptive learning rates
    for each parameter.
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08):
        """
        Initialize Adam.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            beta_1 (float, optional): Exponential decay rate for the
                first moment estimates. Defaults to 0.9.
            beta_2 (float, optional): Exponential decay rate for the
                second moment estimates. Defaults to 0.999.
            epsilon (float, optional): Small value to prevent division by
                zero. Defaults to 1e-08.
        """
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if beta_1 <= 0 or beta_1 >= 1:
            raise ValueError("beta_1 must be in (0, 1)")
        if beta_2 <= 0 or beta_2 >= 1:
            raise ValueError("beta_2 must be in (0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def step(self, w, grad):
        """
        Update the weights using the gradient.

        Args:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss with respect to
                weights.

        Returns:
            np.ndarray: Updated weights.
        """
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
        """
        Reset the optimizer state.
        """
        self.m = None
        self.v = None
        self.t = 0
