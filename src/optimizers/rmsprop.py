import numpy as np


class RMSProp:
    """
    RMSProp optimizer.

    Root Mean Square Propagation.
    """

    def __init__(self, lr=0.001, beta=0.9, epsilon=1e-8):
        """
        Initialize RMSProp.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.001.
            beta (float, optional): Decay rate. Defaults to 0.9.
            epsilon (float, optional): Small value to prevent division by
                zero. Defaults to 1e-8.
        """
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if beta <= 0 or beta >= 1:
            raise ValueError("beta must be in (0, 1)")
        if epsilon <= 0:
            raise ValueError("epsilon must be > 0")

        self.lr = float(lr)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.s = None

    def step(self, w, grad):
        """
        Update the weights using the gradient.

        Args:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss with respect to weights.

        Returns:
            np.ndarray: Updated weights.
        """
        if self.s is None:
            self.s = np.zeros_like(w)

        self.s = self.beta * self.s + (1 - self.beta) * (grad**2)

        return w - self.lr * grad / (np.sqrt(self.s) + self.epsilon)

    def reset(self):
        """
        Reset the optimizer state.
        """
        self.s = None
