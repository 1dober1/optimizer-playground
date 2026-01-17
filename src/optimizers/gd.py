class GD:
    """
    Gradient Descent (GD) optimizer.

    Basic stochastic gradient descent.
    """

    def __init__(self, lr=None):
        """
        Initialize Gradient Descent.

        Args:
            lr (float, optional): Learning rate. Must be greater than 0.
        """
        if lr is None or lr <= 0:
            raise ValueError("lr must be > 0")
        self.lr = lr

    def step(self, w, grad):
        """
        Update the weights using the gradient.

        Args:
            w (np.ndarray): Current weights.
            grad (np.ndarray): Gradient of the loss with respect to weights.

        Returns:
            np.ndarray: Updated weights.
        """
        return w - self.lr * grad
