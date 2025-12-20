import numpy as np

from src.loss import MSE
from src.optimizers.gd import GD
from src.regularizers import Elastic_Net, L2
from src.utils import BatchIterator


class LinearRegression:
    def __init__(
        self,
        fit_intercept=True,
        loss=None,
        reg=None,
        opt=None,
        steps=1000,
        random_state=None,
        batch_size=None,
        loss_smoothing=0.9,
        solver="iterative",
    ) -> None:
        self.fit_intercept = fit_intercept
        self.reg = reg
        self.solver = solver
        self.steps = steps
        self.batch_size = batch_size
        self.rng = np.random.default_rng(random_state)
        self.loss_smoothing = loss_smoothing

        if loss is None:
            self.loss = MSE()
        else:
            self.loss = loss

        if solver == "iterative":
            if opt is None:
                self.opt = GD(lr=0.01)
            else:
                self.opt = opt
        else:
            self.opt = None

        self.batch_iterator = BatchIterator(batch_size, random_state)
        self.w = None

    def fit(self, X, y):
        self.history = []

        X_b = np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X

        if self.solver == "closed":
            self._fit_closed(X_b, y)
        elif self.solver == "iterative":
            self._fit_iterative(X_b, y)
        else:
            raise ValueError(f"Unknown solver: {self.solver}")

        return self

    def _fit_closed(self, X_b, y):
        if self.opt is not None:
            print("Info: Optimizer is ignored for solver='closed'")

        if self.reg is None:
            self.w = np.linalg.pinv(X_b) @ y
            return self

        is_ridge = isinstance(self.reg, L2) or (
            isinstance(self.reg, Elastic_Net) and self.reg.l1_ratio == 0
        )

        if is_ridge:
            I = np.eye(X_b.shape[1])
            if self.fit_intercept:
                I[0, 0] = 0.0

            A = X_b.T @ X_b + (self.reg.alpha * X_b.shape[0]) * I
            b = X_b.T @ y
            self.w = np.linalg.solve(A, b)
            return self

        raise NotImplementedError(
            "Analytical solution is only available for None (OLS) or "
            "L2/Ridge. For L1/ElasticNet use solver='iterative'."
        )

    def _fit_iterative(self, X_b, y):
        if self.steps is None or self.steps <= 0:
            raise ValueError("Steps must be > 0")

        self.w = self.rng.standard_normal(X_b.shape[1]) * 0.01

        is_full_batch = self.batch_iterator.batch_size is None
        if self.loss_smoothing is None:
            current_loss_smoothing = 1.0 if is_full_batch else 0.1
        else:
            current_loss_smoothing = (
                1.0 if is_full_batch else self.loss_smoothing
            )

        Qe = self.loss(y, X_b @ self.w)

        if hasattr(self.opt, "reset"):
            self.opt.reset()

        for _ in range(self.steps):
            X_batch, y_batch = self.batch_iterator.get_batch(X_b, y)

            pred = X_batch @ self.w
            loss_val = self.loss(y_batch, pred)

            if self.reg is not None:
                w_reg = self.w[1:] if self.fit_intercept else self.w
                loss_val += self.reg(w_reg)

            Qe = (
                current_loss_smoothing * loss_val
                + (1 - current_loss_smoothing) * Qe
            )

            grad = self.loss.gradient(X_batch, self.w, y_batch)

            if self.reg is not None and hasattr(self.reg, "grad"):
                if self.fit_intercept:
                    grad[1:] += self.reg.grad(self.w[1:])
                else:
                    grad += self.reg.grad(self.w)

            self.w = self.opt.step(self.w, grad)

            if self.reg is not None and hasattr(self.reg, "prox"):
                lr = self.opt.lr
                if self.fit_intercept:
                    self.w[1:] = self.reg.prox(self.w[1:], lr)
                else:
                    self.w = self.reg.prox(self.w, lr)

            self.history.append(Qe)

    def predict(self, X):
        if self.w is None:
            raise ValueError("Cannot call predict() before fit()")

        X_b = np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X
        return X_b @ self.w
