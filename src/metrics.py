import numpy as np
from src.losses import MAE, MSE, RMSE


def mean_squared_error(y_true, y_pred):
    return MSE()(y_true, y_pred)


def root_mean_squared_error(y_true, y_pred):
    return RMSE()(y_true, y_pred)


def mean_absolute_error(y_true, y_pred):
    return MAE()(y_true, y_pred)


def r2_score(y_true, y_pred):
    return 1 - (
        np.sum((y_true - y_pred) ** 2)
        / np.sum((y_true - np.mean(y_true)) ** 2)
    )


def mean_absolute_percentage_error(y_true, y_pred):
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
