import numpy as np
from src.models import LinearRegression
from src.loss import MSE
from src.optimizers.gd import GD


def test_fit_intercept_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)

    m1 = LinearRegression(
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

    m2 = LinearRegression(
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
