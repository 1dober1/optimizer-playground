import numpy as np

from src.losses import MSE, LogLoss, CrossEntropyLoss
from src.optimizers.gd import GD
from src.regularizers import Elastic_Net, L2
from src.utils import BatchIterator


class LinearRegression:
    """
    Linear Regression model.

    Can be trained using closed-form solution or iterative optimization.
    """

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
        """
        Initialize the Linear Regression model.

        Args:
            fit_intercept (bool, optional): Whether to calculate the
                intercept for this model. Defaults to True.
            loss (object, optional): Loss function. Defaults to None
                (MSE).
            reg (object, optional): Regularizer. Defaults to None.
            opt (object, optional): Optimizer. Defaults to None (GD if
                solver is iterative).
            steps (int, optional): Number of iteration steps. Defaults to
                1000.
            random_state (int, optional): Seed for random number
                generator. Defaults to None.
            batch_size (int, optional): Size of the batch for iterative
                optimization. Defaults to None (full batch).
            loss_smoothing (float, optional): Smoothing factor for loss
                history. Defaults to 0.9.
            solver (str, optional): Solver to use, 'closed' or
                'iterative'. Defaults to "iterative".
        """
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
        """
        Fit the linear regression model.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.

        Returns:
            LinearRegression: The fitted model.
        """
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
        """
        Fit using the closed-form (analytical) solution.

        Args:
            X_b (np.ndarray): Feature matrix with bias term if
                applicable.
            y (np.ndarray): Target values.
        """
        if self.opt is not None:
            print("Info: Optimizer is ignored for solver='closed'")

        if self.reg is None:
            self.w = np.linalg.pinv(X_b) @ y
            return self

        is_ridge = isinstance(self.reg, L2) or (
            isinstance(self.reg, Elastic_Net) and self.reg.l1_ratio == 0
        )

        if is_ridge:
            I_mat = np.eye(X_b.shape[1])
            if self.fit_intercept:
                I_mat[0, 0] = 0.0

            A = X_b.T @ X_b + (self.reg.alpha * X_b.shape[0]) * I_mat
            b = X_b.T @ y
            self.w = np.linalg.solve(A, b)
            return self

        raise NotImplementedError(
            "Analytical solution is only available for None (OLS) or "
            "L2/Ridge. For L1/ElasticNet use solver='iterative'."
        )

    def _fit_iterative(self, X_b, y):
        """
        Fit using iterative optimization.

        Args:
            X_b (np.ndarray): Feature matrix with bias term if
                applicable.
            y (np.ndarray): Target values.
        """
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
        if self.reg is not None:
            w_reg = self.w[1:] if self.fit_intercept else self.w
            Qe += self.reg(w_reg)

        if hasattr(self.opt, "reset"):
            self.opt.reset()

        use_decoupled_wd = bool(
            getattr(self.opt, "decoupled_weight_decay", False)
        )

        if use_decoupled_wd and hasattr(self.opt, "exclude_intercept"):
            self.opt.exclude_intercept = self.fit_intercept

        if use_decoupled_wd and self.reg is not None:
            inferred_wd = None
            if isinstance(self.reg, L2):
                inferred_wd = 2.0 * self.reg.alpha
            elif isinstance(self.reg, Elastic_Net):
                inferred_wd = self.reg.alpha * (1.0 - self.reg.l1_ratio)

            if (
                inferred_wd is not None
                and float(getattr(self.opt, "weight_decay", 0.0)) == 0.0
            ):
                self.opt.weight_decay = float(inferred_wd)

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
                skip_l2_grad = use_decoupled_wd and isinstance(
                    self.reg, (L2, Elastic_Net)
                )
                if not skip_l2_grad:
                    if self.fit_intercept:
                        grad[1:] += self.reg.grad(self.w[1:])
                    else:
                        grad += self.reg.grad(self.w)

            self.w = self.opt.step(self.w, grad)

            if self.reg is not None and hasattr(self.reg, "prox"):
                lr = getattr(self.opt, "lr", None)
                if lr is None:
                    raise AttributeError(
                        "Optimizer must have attribute 'lr' to use "
                        "prox-regularization (L1/ElasticNet)."
                    )

                if self.fit_intercept:
                    self.w[1:] = self.reg.prox(self.w[1:], lr)
                else:
                    self.w = self.reg.prox(self.w, lr)

            self.history.append(Qe)

    def predict(self, X):
        """
        Predict using the linear model.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Predicted values.
        """
        if self.w is None:
            raise ValueError("Cannot call predict() before fit()")

        X_b = np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X
        return X_b @ self.w


class LogisticRegression:
    """
    Logistic Regression model.

    Supports binary and multi-class classification.
    """

    def __init__(
        self,
        fit_intercept=True,
        opt=None,
        reg=None,
        batch_size=None,
        loss_smoothing=0.9,
        steps=1000,
        random_state=None,
    ):
        """
        Initialize the Logistic Regression model.

        Args:
            fit_intercept (bool, optional): Whether to calculate the
                intercept for this model. Defaults to True.
            opt (object, optional): Optimizer. Defaults to None (GD).
            reg (object, optional): Regularizer. Defaults to None.
            batch_size (int, optional): Size of the batch for iterative
                optimization. Defaults to None (full batch).
            loss_smoothing (float, optional): Smoothing factor for loss
                history. Defaults to 0.9.
            steps (int, optional): Number of iteration steps. Defaults
                to 1000.
            random_state (int, optional): Seed for random number
                generator. Defaults to None.
        """
        self.fit_intercept = fit_intercept
        self.steps = steps
        self.reg = reg
        self.rng = np.random.default_rng(random_state)
        self.loss_smoothing = loss_smoothing
        self.batch_iterator = BatchIterator(
            batch_size=batch_size, random_state=random_state
        )
        self.w = None

        if opt is None:
            self.opt = GD(lr=0.01)
        else:
            self.opt = opt

    def sigmoid_(self, Z):
        """
        Sigmoid activation function.

        Args:
            Z (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid output.
        """
        z = np.asarray(Z)
        out = np.empty_like(z, dtype=float)
        pos = z >= 0

        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)

        return out

    def softmax_(self, Z):
        """
        Softmax activation function.

        Args:
            Z (np.ndarray): Input values.

        Returns:
            np.ndarray: Softmax output.
        """
        Z = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Z)
        return expZ / expZ.sum(axis=1, keepdims=True)

    def fit(self, X, y):
        """
        Fit the logistic regression model.

        Args:
            X (np.ndarray): Training data.
            y (np.ndarray): Target values.

        Returns:
            LogisticRegression: The fitted model.
        """
        if self.steps is None or self.steps <= 0:
            raise ValueError("Steps must be > 0")

        self.history = []

        X_b = np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X

        classes, y_idx = np.unique(y, return_inverse=True)
        self.classes_ = classes
        n_classes = len(classes)

        if n_classes > 2:
            self.is_multiclass = True
            y_one_hot = np.zeros((y.shape[0], n_classes))
            y_one_hot[np.arange(y.shape[0]), y_idx] = 1
            self.w = self.rng.standard_normal((X_b.shape[1], n_classes)) * 0.01
            self.loss = CrossEntropyLoss()
        else:
            self.is_multiclass = False
            y_one_hot = y_idx
            self.w = self.rng.standard_normal(X_b.shape[1]) * 0.01
            self.loss = LogLoss()

        is_full_batch = self.batch_iterator.batch_size is None
        if self.loss_smoothing is None:
            current_loss_smoothing = 1.0 if is_full_batch else 0.1
        else:
            current_loss_smoothing = (
                1.0 if is_full_batch else self.loss_smoothing
            )

        Qe = self.loss(y_one_hot, X_b @ self.w)
        if self.reg is not None:
            w_reg = self.w[1:] if self.fit_intercept else self.w
            Qe += self.reg(w_reg)

        if hasattr(self.opt, "reset"):
            self.opt.reset()

        use_decoupled_wd = bool(
            getattr(self.opt, "decoupled_weight_decay", False)
        )

        if use_decoupled_wd and hasattr(self.opt, "exclude_intercept"):
            self.opt.exclude_intercept = self.fit_intercept

        if use_decoupled_wd and self.reg is not None:
            inferred_wd = None
            if isinstance(self.reg, L2):
                inferred_wd = 2.0 * self.reg.alpha
            elif isinstance(self.reg, Elastic_Net):
                inferred_wd = self.reg.alpha * (1.0 - self.reg.l1_ratio)

            if (
                inferred_wd is not None
                and float(getattr(self.opt, "weight_decay", 0.0)) == 0.0
            ):
                self.opt.weight_decay = float(inferred_wd)

        for _ in range(self.steps):
            X_batch, y_batch = self.batch_iterator.get_batch(X_b, y_one_hot)

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
                skip_l2_grad = use_decoupled_wd and isinstance(
                    self.reg, (L2, Elastic_Net)
                )
                if not skip_l2_grad:
                    if self.fit_intercept:
                        grad[1:] += self.reg.grad(self.w[1:])
                    else:
                        grad += self.reg.grad(self.w)

            self.w = self.opt.step(self.w, grad)

            if self.reg is not None and hasattr(self.reg, "prox"):
                lr = getattr(self.opt, "lr", None)
                if lr is None:
                    raise AttributeError(
                        "Optimizer must have attribute 'lr' to use "
                        "prox-regularization (L1/ElasticNet)."
                    )

                if self.fit_intercept:
                    self.w[1:] = self.reg.prox(self.w[1:], lr)
                else:
                    self.w = self.reg.prox(self.w, lr)

            self.history.append(Qe)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Returns the probability of the sample for each
                class in the model.
        """
        if self.w is None:
            raise ValueError("Cannot call predict() before fit()")

        X_b = np.c_[np.ones((X.shape[0], 1)), X] if self.fit_intercept else X

        Z = X_b @ self.w

        if self.is_multiclass:
            return self.softmax_(Z)

        p1 = self.sigmoid_(Z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Args:
            X (np.ndarray): Samples.

        Returns:
            np.ndarray: Predicted class label per sample.
        """
        proba = self.predict_proba(X)

        if self.is_multiclass:
            idx = np.argmax(proba, axis=1)
            return self.classes_[idx]

        return np.where(proba[:, 1] >= 0.5, self.classes_[1], self.classes_[0])
