import numpy as np


class L2:
    """
    L2 Regularization (Ridge).

    Adds a penalty equal to the square of the magnitude of coefficients.
    """

    def __init__(self, alpha):
        """
        Initialize the L2 regularizer.

        Args:
            alpha (float): Regularization strength.
        """
        self.alpha = float(alpha)

    def __call__(self, w):
        """
        Calculate the L2 regularization penalty.

        Args:
            w (np.ndarray): Weights.

        Returns:
            float: Penalty value.
        """
        return self.alpha * np.sum(np.square(w))

    def grad(self, w):
        """
        Compute the gradient of the L2 penalty.

        Args:
            w (np.ndarray): Weights.

        Returns:
            np.ndarray: Gradient of the penalty.
        """
        return 2 * self.alpha * w


class L1:
    """
    L1 Regularization (Lasso).

    Adds a penalty equal to the absolute value of the magnitude of
    coefficients.
    """

    def __init__(self, alpha):
        """
        Initialize the L1 regularizer.

        Args:
            alpha (float): Regularization strength.
        """
        self.alpha = float(alpha)

    def __call__(self, w):
        """
        Calculate the L1 regularization penalty.

        Args:
            w (np.ndarray): Weights.

        Returns:
            float: Penalty value.
        """
        return self.alpha * np.sum(np.abs(w))

    def prox(self, w, lr: float):
        """
        Proximal operator for L1 regularization.

        Args:
            w (np.ndarray): Weights.
            lr (float): Learning rate.

        Returns:
            np.ndarray: Updated weights after applying the proximal
                operator.
        """
        t = lr * self.alpha
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)


class Elastic_Net:
    """
    Elastic-Net Regularization.

    Linearly combines the L1 and L2 penalties.
    """

    def __init__(self, alpha, l1_ratio):
        """
        Initialize the Elastic-Net regularizer.

        Args:
            alpha (float): Regularization strength.
            l1_ratio (float): The ElasticNet mixing parameter, with
                0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an
                L2 penalty. For l1_ratio = 1 it is an L1 penalty.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, w):
        """
        Calculate the Elastic-Net regularization penalty.

        Args:
            w (np.ndarray): Weights.

        Returns:
            float: Penalty value.
        """
        return self.alpha * self.l1_ratio * np.sum(
            np.abs(w)
        ) + 0.5 * self.alpha * (1 - self.l1_ratio) * np.sum(np.square(w))

    def grad(self, w):
        """
        Compute the gradient of the L2 part of Elastic-Net.

        Args:
            w (np.ndarray): Weights.

        Returns:
            np.ndarray: Gradient of the L2 (smooth) part.
        """
        return self.alpha * (1 - self.l1_ratio) * w

    def prox(self, w, lr: float):
        """
        Proximal operator for the L1 part of Elastic-Net.

        Args:
            w (np.ndarray): Weights.
            lr (float): Learning rate.

        Returns:
            np.ndarray: Updated weights after applying the proximal
                operator.
        """
        t = lr * self.alpha * self.l1_ratio
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)
