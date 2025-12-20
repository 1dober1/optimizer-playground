import argparse

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression as SklearnLR,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.loss import MSE
from src.models import LinearRegression
from src.optimizers.adam import Adam
from src.optimizers.gd import GD
from src.regularizers import Elastic_Net, L1, L2


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Linear Regression model"
    )

    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps"
    )

    parser.add_argument(
        "--reg",
        choices=["none", "l1", "l2", "elastic"],
        default="none",
        help="Regularization type",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Regularization strength",
    )
    parser.add_argument(
        "--opt",
        choices=["gd", "adam"],
        default=None,
        help="Optimizer type",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--l1_ratio",
        type=float,
        default=0.5,
        help="L1 ratio for Elastic-Net",
    )

    parser.add_argument(
        "--fit_intercept", action="store_true", help="Fit intercept"
    )
    parser.add_argument(
        "--no_intercept",
        dest="fit_intercept",
        action="store_false",
        help="Do not fit intercept",
    )
    parser.set_defaults(fit_intercept=True)

    parser.add_argument(
        "--solver",
        default="iterative",
        help="Solver for Linear Regression",
    )

    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples"
    )
    parser.add_argument(
        "--n_features", type=int, default=50, help="Number of features"
    )
    parser.add_argument(
        "--noise", type=float, default=10.0, help="Noise level"
    )

    return parser.parse_args()


def main():
    args = get_args()

    X, y = make_regression(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=5,
        noise=args.noise,
        random_state=args.seed,
    )

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    regularizer = None
    if args.reg == "l1":
        regularizer = L1(alpha=args.alpha)
    elif args.reg == "l2":
        regularizer = L2(alpha=args.alpha)
    elif args.reg == "elastic":
        regularizer = Elastic_Net(alpha=args.alpha, l1_ratio=args.l1_ratio)

    optimizer = None
    if args.opt == "gd":
        optimizer = GD(lr=args.lr)
    elif args.opt == "adam":
        optimizer = Adam(lr=args.lr)

    if args.solver == "iterative" and optimizer is None:
        print(
            "Warning: optimizer not specified for iterative solver. "
            "Defaulting to GD."
        )
        optimizer = GD(lr=args.lr)
        args.opt = "gd (auto)"

    print(
        "\nTraining model with "
        f"lr={args.lr}, reg={args.reg}, steps={args.steps}, "
        f"opt={args.opt}, batch_size={args.batch_size}, "
        f"alpha={args.alpha}, solver={args.solver}"
    )

    model = LinearRegression(
        fit_intercept=args.fit_intercept,
        loss=MSE(),
        reg=regularizer,
        opt=optimizer,
        steps=args.steps,
        random_state=args.seed,
        batch_size=args.batch_size,
        solver=args.solver,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_mse = MSE()(y_train, y_pred_train)
    test_mse = MSE()(y_test, y_pred_test)

    print("=" * 100 + "\n")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    if args.reg in ("l1", "elastic"):
        w = model.w[1:] if args.fit_intercept else model.w
        print(f"Zero weights count: {np.sum(np.abs(w) < 1e-8)}")

    print("\n" + "=" * 43 + " SANITY CHECK " + "=" * 43 + "\n")

    n_samples = X_train.shape[0]

    if args.reg == "none":
        sk_model = SklearnLR(fit_intercept=args.fit_intercept)
    elif args.reg == "l2":
        sk_model = Ridge(
            alpha=args.alpha * n_samples,
            fit_intercept=args.fit_intercept,
            max_iter=args.steps,
            random_state=args.seed,
        )
    elif args.reg == "l1":
        sk_model = Lasso(
            alpha=args.alpha / 2.0,
            fit_intercept=args.fit_intercept,
            max_iter=args.steps,
            random_state=args.seed,
        )
    else:
        sk_model = ElasticNet(
            alpha=args.alpha / 2.0,
            l1_ratio=args.l1_ratio,
            fit_intercept=args.fit_intercept,
            max_iter=args.steps,
            random_state=args.seed,
        )

    sk_model.fit(X_train, y_train)
    sk_pred = sk_model.predict(X_test)
    sk_mse = MSE()(y_test, sk_pred)

    print(f"My Model MSE:      {test_mse:.6f}")
    print(f"Sklearn Model MSE: {sk_mse:.6f}")

    diff = abs(test_mse - sk_mse)
    print(f"Difference:        {diff:.6f}")


if __name__ == "__main__":
    main()
