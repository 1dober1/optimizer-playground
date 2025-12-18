import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from src.models import LinearRegression
from src.loss import mean_squared_error
from src.optimizers.gd import GD


X, y = make_regression(n_samples=15, n_features=3, noise=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

gd_opt = GD(lr=0.1)
lin_reg = LinearRegression(opt=gd_opt, steps=1000)

lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print(f"Predicted values: {y_pred}")
print(f"Actual values: {y_test}")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
