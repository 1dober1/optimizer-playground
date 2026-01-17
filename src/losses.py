import numpy as np


class MSE:
    """
    Mean Squared Error (MSE) loss function.

    Computes the average squared difference between the estimated values
    and the actual value.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the MSE loss.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Content of the loss function.
        """
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X, w, y):
        """
        Compute the gradient of the MSE loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y (np.ndarray): True target values.

        Returns:
            np.ndarray: Gradient vector.
        """
        y_pred = X @ w
        error = y_pred - y.flatten()
        return 2 / X.shape[0] * (X.T @ error)


class RMSE:
    """
    Root Mean Squared Error (RMSE) loss function.

    Computes the square root of the average squared difference between
    the estimated values and the actual value.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the RMSE loss.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Content of the loss function.
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def gradient(self, X, w, y, eps=1e-12):
        """
        Compute the gradient of the RMSE loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y (np.ndarray): True target values.
            eps (float, optional): Small epsilon to avoid division by
                zero. Defaults to 1e-12.

        Returns:
            np.ndarray: Gradient vector.
        """
        y_pred = X @ w
        error = y_pred - y
        rmse = np.sqrt(np.mean(error**2)) + eps
        return (X.T @ error) / (X.shape[0] * rmse)


class MAE:
    """
    Mean Absolute Error (MAE) loss function.

    Computes the average absolute difference between the estimated values
    and the actual value.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the MAE loss.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Content of the loss function.
        """
        return np.mean(np.abs(y_true - y_pred))

    def gradient(self, X, w, y):
        """
        Compute the gradient of the MAE loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y (np.ndarray): True target values.

        Returns:
            np.ndarray: Gradient vector.
        """
        y_pred = X @ w
        error = y_pred - y
        return (X.T @ np.sign(error)) / X.shape[0]


class Huber:
    """
    Huber loss function.

    A loss function used in robust regression, that is less sensitive to
    outliers in data than the squared error loss.
    """

    def __init__(self, delta=1.0):
        """
        Initialize the Huber loss.

        Args:
            delta (float, optional): The point where the Huber loss
                function changes from a quadratic to linear. Defaults to
                1.0.
        """
        self.delta = delta

    def __call__(self, y_true, y_pred):
        """
        Calculate the Huber loss.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Content of the loss function.
        """
        error = y_true - y_pred
        is_small_error = np.abs(error) <= self.delta

        squared_loss = 0.5 * error**2
        linear_loss = self.delta * (np.abs(error) - self.delta / 2)

        return np.mean(np.where(is_small_error, squared_loss, linear_loss))

    def gradient(self, X, w, y):
        """
        Compute the gradient of the Huber loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y (np.ndarray): True target values.

        Returns:
            np.ndarray: Gradient vector.
        """
        y_pred = X @ w
        error = y_pred - y
        is_small_error = np.abs(error) <= self.delta
        grad_error = np.where(
            is_small_error, error, self.delta * np.sign(error)
        )
        return (X.T @ grad_error) / X.shape[0]


class LogCosh:
    """
    Log-Cosh loss function.

    Logarithm of the hyperbolic cosine of the prediction error.
    """

    def __call__(self, y_true, y_pred):
        """
        Calculate the Log-Cosh loss.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            float: Content of the loss function.
        """
        return np.mean(np.log(np.cosh(y_true - y_pred)))

    def gradient(self, X, w, y):
        """
        Compute the gradient of the Log-Cosh loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y (np.ndarray): True target values.

        Returns:
            np.ndarray: Gradient vector.
        """
        y_pred = X @ w
        error = y_pred - y
        return (X.T @ np.tanh(error)) / X.shape[0]


class LogLoss:
    """
    Log Loss (Binary Cross-Entropy) function.

    Used for binary classification tasks.
    """

    def __call__(self, y_true, y_logits):
        """
        Calculate the Log Loss.

        Args:
            y_true (np.ndarray): True binary labels.
            y_logits (np.ndarray): Predicted logits.

        Returns:
            float: Content of the loss function.
        """
        return np.mean(np.logaddexp(0, y_logits) - y_true * y_logits)

    def gradient(self, X, w, y_true):
        """
        Compute the gradient of the Log Loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight vector.
            y_true (np.ndarray): True binary labels.

        Returns:
            np.ndarray: Gradient vector.
        """
        z = X @ w

        p = np.empty_like(z, dtype=float)
        pos = z >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        p[~pos] = ez / (1.0 + ez)

        error = p - y_true
        return (X.T @ error) / X.shape[0]


class CrossEntropyLoss:
    """
    Cross-Entropy Loss function.

    Used for multi-class classification tasks.
    """

    def __call__(self, y_true, y_logits):
        """
        Calculate the Cross-Entropy Loss.

        Args:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_logits (np.ndarray): Predicted logits.

        Returns:
            float: Content of the loss function.
        """
        logits_max = np.max(y_logits, axis=1, keepdims=True)
        logits_shifted = y_logits - logits_max

        exp_logits = np.exp(logits_shifted)
        log_sum_exp = np.log(np.sum(exp_logits, axis=1, keepdims=True))
        log_probs = logits_shifted - log_sum_exp

        return -np.mean(np.sum(y_true * log_probs, axis=1))

    def gradient(self, X, w, y_true):
        """
        Compute the gradient of the Cross-Entropy Loss with respect to weights.

        Args:
            X (np.ndarray): Feature matrix.
            w (np.ndarray): Weight matrix.
            y_true (np.ndarray): True labels (one-hot encoded).

        Returns:
            np.ndarray: Gradient matrix.
        """
        logits = X @ w
        logits_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        error = probs - y_true

        return (X.T @ error) / X.shape[0]
