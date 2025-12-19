import numpy as np


class BatchIterator:
    def __init__(self, batch_size=None, random_state=None):
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)

    def get_batch(self, X, y):
        if self.batch_size is None:
            return X, y
        idx = self.rng.integers(low=0, high=X.shape[0], size=self.batch_size)
        return X[idx], y[idx]
