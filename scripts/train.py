import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models import LinearRegression
from src.optimizers.gd import GD
from src.loss import MSE
from src.regularizers import L1, L2


X, y = make_regression(n_samples=200, n_features=1, noise=10.0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression(
    fit_intercept=True,
    loss=MSE(),
    reg=L1(alpha=0.1),
    opt=GD(lr=0.3),
    steps=1000,
)


model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"MSE: {MSE()(y_test, y_pred)}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

X_line = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 200)
y_line = model.predict(X_line)

# ax1.scatter(X_train[:, 0], y_train, color="blue", alpha=0.7)
ax1.scatter(X_test[:, 0], y_test, color="red", alpha=0.7)
ax1.plot(X_line, y_line)

ax2.plot(model.history, color="red")
ax2.set_xlabel("Steps")
ax2.set_ylabel("Loss")

plt.show()
