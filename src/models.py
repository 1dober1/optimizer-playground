import numpy as np


class LinearRegression:
    def __init__(
        self, fit_intercept=True, loss=None, penalty=None, opt=None, steps=None
    ) -> None:
        if loss is None:
            raise Exception("Loss function cannot be empty")
        if opt is None:
            raise Exception("Optimizer cannot be empty")
        if steps is None or steps < 0:
            raise ValueError("Steps must be > 0")

        self.fit_intercept = fit_intercept
        self.loss = loss
        self.reg = penalty
        self.opt = opt
        self.w = None

    def fit(self, X, y):
        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X

        self.w = np.random.randn(X_b.shape[1]) * 0.01

        for _ in range(self.steps):
            grad = self.loss.gradient(X_b, self.w, y)
            self.w = self.opt.step(self.w, grad)

        return self

    def predict(self, X):
        if self.w is not None:
            if self.fit_intercept:
                X_b = np.c_[np.ones((X.shape[0], 1)), X]
            else:
                X_b = X
            return X_b @ self.w
        else:
            raise Exception(f"Cannot call predict() before fit()")
