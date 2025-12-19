import numpy as np

from src.utils import BatchIterator


class LinearRegression:
    def __init__(
        self,
        fit_intercept=True,
        loss=None,
        reg=None,
        opt=None,
        steps=None,
        random_state=None,
        batch_size=None,
        lmd=None,
    ) -> None:
        if loss is None:
            raise Exception("Loss function cannot be empty")
        if opt is None:
            raise Exception("Optimizer cannot be empty")
        if steps is None or steps <= 0:
            raise ValueError("Steps must be > 0")

        self.fit_intercept = fit_intercept
        self.loss = loss
        self.reg = reg
        self.opt = opt
        self.w = None
        self.steps = steps
        self.rng = np.random.default_rng(random_state)
        self.batch_iterator = BatchIterator(batch_size, random_state)
        self.lmd = lmd

    def fit(self, X, y):
        self.history = []

        if self.fit_intercept:
            X_b = np.c_[np.ones((X.shape[0], 1)), X]
        else:
            X_b = X

        self.w = self.rng.standard_normal(X_b.shape[1]) * 0.01

        is_full_batch = self.batch_iterator.batch_size is None
        if self.lmd is None:
            current_lmd = 1.0 if is_full_batch else 0.1
        else:
            current_lmd = 1.0 if is_full_batch else self.lmd

        Qe = self.loss(y, X_b @ self.w)

        if hasattr(self.opt, "reset"):
            self.opt.reset()

        for _ in range(self.steps):
            X_batch, y_batch = self.batch_iterator.get_batch(X_b, y)

            pred = X_batch @ self.w
            loss_val = self.loss(y_batch, pred)
            if self.reg is not None:
                loss_val += self.reg(
                    self.w[1:] if self.fit_intercept else self.w
                )
            Qe = current_lmd * loss_val + (1 - current_lmd) * Qe

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
