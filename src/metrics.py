import numpy as np
from src.losses import MAE, MSE, RMSE


def mean_squared_error(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: MSE value.
    """
    return MSE()(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: RMSE value.
    """
    return RMSE()(y_true, y_pred)


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: MAE value.
    """
    return MAE()(y_true, y_pred)


def r2_score(y_true, y_pred):
    """
    Calculate the R-squared score.

    The coefficient of determination regression score function.

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: R2 score.
    """
    return 1 - (
        np.sum((y_true - y_pred) ** 2)
        / np.sum((y_true - np.mean(y_true)) ** 2)
    )


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).

    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.

    Returns:
        float: MAPE value in percentage.
    """
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
