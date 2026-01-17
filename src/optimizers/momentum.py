import numpy as np


class Momentum:
    """
    Momentum optimizer.

    Accelerates SGD in the relevant direction and dampens oscillations.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        Initialize Momentum.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.9.
        """
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")

        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, w, grad):
        """
        Update the weights using the gradient.

        Args:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss with respect to weights.

        Returns:
            np.ndarray: Updated weights.
        """
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v + self.lr * grad

        return w - self.v

    def reset(self):
        """
        Reset the optimizer state.
        """
        self.v = None


class NesterovMomentum:
    """
    Nesterov Momentum optimizer.

    A variant of momentum that uses the 'lookahead' position of the
    parameters.
    """

    def __init__(self, lr=0.01, momentum=0.9):
        """
        Initialize Nesterov Momentum.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.9.
        """
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        if momentum < 0 or momentum >= 1:
            raise ValueError("momentum must be in [0, 1)")

        self.lr = lr
        self.momentum = momentum
        self.v = None

    def step(self, w, grad):
        """
        Update the weights using the gradient.

        Args:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss with respect to weights.

        Returns:
            np.ndarray: Updated weights.
        """
        if self.v is None:
            self.v = np.zeros_like(w)

        self.v = self.momentum * self.v + self.lr * grad

        return w - (self.momentum * self.v + self.lr * grad)

    def reset(self):
        """
        Reset the optimizer state.
        """
        self.v = None
