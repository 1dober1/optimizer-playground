import numpy as np


class BatchIterator:
    """
    Iterator for generating batches of data.
    """

    def __init__(self, batch_size=None, random_state=None):
        """
        Initialize the BatchIterator.

        Args:
            batch_size (int, optional): Size of the batch. If None,
                returns the full dataset. Defaults to None.
            random_state (int, optional): Seed for random number
                generator. Defaults to None.
        """
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)

    def get_batch(self, X, y):
        """
        Get a random batch of data.

        Args:
            X (np.ndarray): Samples.
            y (np.ndarray): Target values.

        Returns:
            tuple: (X_batch, y_batch)
        """
        if self.batch_size is None:
            return X, y
        idx = self.rng.integers(low=0, high=X.shape[0], size=self.batch_size)
        return X[idx], y[idx]
