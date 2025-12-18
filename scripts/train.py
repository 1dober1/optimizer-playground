from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models import LinearRegression
from src.optimizers.gd import GD
from src.loss import MSE


X, y = make_regression(n_samples=200, n_features=3, noise=1.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression(
    fit_intercept=True,
    loss=MSE(),
    opt=GD(lr=0.1),
    steps=1000,
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {MSE()(y_test, y_pred)}")
