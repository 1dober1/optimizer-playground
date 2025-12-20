import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as sklearn_model, Ridge
from src.models import LinearRegression as my_model
from src.losses import MSE
from src.optimizers.gd import GD
from src.regularizers import L2


def test_fit_intercept_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    m1 = my_model(
        fit_intercept=True,
        loss=MSE(),
        opt=GD(lr=0.01),
        steps=5,
        random_state=0,
    )
    m1.fit(X, y)
    assert m1.w.shape == (5,)
    pred1 = m1.predict(X)
    assert pred1.shape == (30,)

    m2 = my_model(
        fit_intercept=False,
        loss=MSE(),
        opt=GD(lr=0.01),
        steps=5,
        random_state=0,
    )
    m2.fit(X, y)
    assert m2.w.shape == (4,)
    pred2 = m2.predict(X)
    assert pred2.shape == (30,)


def test_closed_form_sanity_no_reg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 15))
    y = rng.normal(size=200)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    my_model_no_reg = my_model(fit_intercept=True, loss=MSE(), solver="closed")
    my_model_no_reg.fit(X_train, y_train)

    sklearn_model_no_reg = sklearn_model(fit_intercept=True)
    sklearn_model_no_reg.fit(X_train, y_train)

    my_bias = my_model_no_reg.w[0]
    sk_bias = sklearn_model_no_reg.intercept_
    assert np.abs(my_bias - sk_bias) < 1e-4

    my_coefs = my_model_no_reg.w[1:]
    sk_coefs = sklearn_model_no_reg.coef_
    assert np.allclose(my_coefs, sk_coefs, atol=1e-4)

    my_pred = my_model_no_reg.predict(X_test)
    sk_pred = sklearn_model_no_reg.predict(X_test)
    assert np.allclose(my_pred, sk_pred, atol=1e-4)


def test_closed_form_l2_sanity():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 10))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.5, size=100)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    n_samples = X_train.shape[0]

    alpha_val = 0.1

    my_ridge = my_model(
        loss=MSE(),
        solver="closed",
        reg=L2(alpha=alpha_val),
        fit_intercept=True,
    )
    my_ridge.fit(X_train, y_train)

    sk_ridge = Ridge(alpha=alpha_val * n_samples, fit_intercept=True)
    sk_ridge.fit(X_train, y_train)

    assert np.isclose(my_ridge.w[0], sk_ridge.intercept_, atol=1e-4)
    assert np.allclose(my_ridge.w[1:], sk_ridge.coef_, atol=1e-4)

    my_pred = my_ridge.predict(X_test)
    sk_pred = sk_ridge.predict(X_test)
    assert np.allclose(my_pred, sk_pred, atol=1e-4)
