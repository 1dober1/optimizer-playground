import numpy as np
from src.loss import grad_mean_squared_error


class LinearRegression:
    def __init__(self, loss=None, penalty=None, opt=None, steps=None) -> None:
        self.loss = loss
        self.reg = penalty
        self.opt = opt
        self.w = None
        self.steps = steps

    def fit(self, X, y):
        self.w = np.zeros(shape=X.shape[1])
        for _ in range(self.steps):
            grad = grad_mean_squared_error(X, self.w, y)
            self.w = self.opt.step(self.w, grad)

    def predict(self, X_test):
        return X_test @ self.w

    def _loss(self):
        pass

    def _gradient(self):
        pass
