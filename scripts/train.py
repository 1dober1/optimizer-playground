import os
import csv
import time
import argparse
from datetime import datetime

import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression as SklearnLR,
    LogisticRegression as SklearnLogReg,
    Ridge,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.losses import MSE, RMSE, MAE, Huber, LogCosh
from src.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from src.models import LinearRegression, LogisticRegression
from src.optimizers.adam import Adam
from src.optimizers.gd import GD
from src.regularizers import Elastic_Net, L1, L2


def get_args():
    parser = argparse.ArgumentParser(
        description="Train Linear Regression model"
    )

    parser.add_argument(
        "--model",
        choices=["linear", "logistic"],
        default="linear",
        help="Model type: linear regression or logistic regression",
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
        "--loss",
        type=str,
        choices=["mse", "rmse", "mae", "huber", "logcosh"],
        default="mse",
        help="Loss Function",
    )

    parser.add_argument("--delta", type=float, default=1.0, help="Huber delta")

    parser.add_argument(
        "--n_samples", type=int, default=200, help="Number of samples"
    )
    parser.add_argument(
        "--n_features", type=int, default=50, help="Number of features"
    )
    parser.add_argument(
        "--noise", type=float, default=10.0, help="Noise level"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        help="Number of classes for logistic regression",
    )

    return parser.parse_args()


def log_run(path, row: dict):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = get_args()

    if args.model == "linear":
        X, y = make_regression(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=5,
            noise=args.noise,
            random_state=args.seed,
        )
    else:
        X, y = make_classification(
            n_samples=args.n_samples,
            n_features=args.n_features,
            n_informative=5,
            n_redundant=5,
            n_classes=args.n_classes,
            random_state=args.seed,
        )

    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    if args.loss == "mse":
        loss_function = MSE()
    elif args.loss == "rmse":
        loss_function = RMSE()
    elif args.loss == "mae":
        loss_function = MAE()
    elif args.loss == "huber":
        loss_function = Huber(delta=args.delta)
    elif args.loss == "logcosh":
        loss_function = LogCosh()
    else:
        loss_function = None

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

    if args.model == "linear":
        if args.solver == "iterative" and optimizer is None:
            print(
                "Warning: optimizer not specified for iterative solver. "
                "Defaulting to GD."
            )
            optimizer = GD(lr=args.lr)
            args.opt = "gd (auto)"

        if args.solver == "closed" and args.reg in ("l1", "elastic"):
            if args.reg == "elastic" and args.l1_ratio == 0:
                pass
            else:
                print(
                    f"Warning: solver='closed' does not support "
                    f"reg='{args.reg}'. Switching to solver='iterative'."
                )
                args.solver = "iterative"
                if optimizer is None:
                    optimizer = GD(lr=args.lr)
                    args.opt = "gd (auto)"

        if (
            args.loss in ("rmse", "mae", "huber", "logcosh")
            and args.lr <= 0.01
        ):
            print(
                f"Info: loss='{args.loss}' may require higher learning rate "
                f"(e.g., --lr 0.1) or more steps for convergence."
            )
    else:
        if optimizer is None:
            print("Warning: optimizer not specified. Defaulting to GD.")
            optimizer = GD(lr=args.lr)
            args.opt = "gd (auto)"

        if args.loss != "mse":
            print(
                f"Info: --loss='{args.loss}' is ignored for logistic "
                "regression. Using LogLoss or CrossEntropyLoss."
            )

    solver_str = args.solver if args.model == "linear" else "N/A"
    print(
        f"\nTraining {args.model} regression model with "
        f"loss={args.loss}, lr={args.lr}, reg={args.reg}, "
        f"steps={args.steps}, opt={args.opt}, "
        f"batch_size={args.batch_size}, alpha={args.alpha}, "
        f"solver={solver_str}"
    )

    if args.model == "linear":
        model = LinearRegression(
            fit_intercept=args.fit_intercept,
            loss=loss_function,
            reg=regularizer,
            opt=optimizer,
            steps=args.steps,
            random_state=args.seed,
            batch_size=args.batch_size,
            solver=args.solver,
        )
    else:
        model = LogisticRegression(
            fit_intercept=args.fit_intercept,
            reg=regularizer,
            opt=optimizer,
            steps=args.steps,
            random_state=args.seed,
            batch_size=args.batch_size,
        )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    if args.model == "linear":
        y_pred_test = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred_test)
        rmse = root_mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        mape = mean_absolute_percentage_error(y_test, y_pred_test)

        w = model.w[1:] if args.fit_intercept else model.w
        zero_count = int(np.sum(np.abs(w) < 1e-8))
        sparsity = float(zero_count / w.size)
        final_train_loss = model.history[-1] if model.history else None

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "solver": args.solver,
            "opt": args.opt,
            "lr": args.lr,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "loss": args.loss,
            "reg": args.reg,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "fit_intercept": args.fit_intercept,
            "train_time_sec": train_time,
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2_score": r2,
            "mape": mape,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "zero_weights": zero_count,
            "sparsity": sparsity,
            "final_train_loss": final_train_loss,
        }
        log_run("./logs/runs.csv", row)

        print("=" * 100 + "\n")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R2: {r2:.4f}")
        print(f"MAPE: {mape:.4f}")

        if args.reg in ("l1", "elastic"):
            print(f"Zero weights count: {zero_count}, sparsity: {sparsity}")

    else:
        y_pred_test = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred_test)

        if args.n_classes == 2:
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test)
        else:
            precision = precision_score(y_test, y_pred_test, average="macro")
            recall = recall_score(y_test, y_pred_test, average="macro")
            f1 = f1_score(y_test, y_pred_test, average="macro")

        w = model.w[1:] if args.fit_intercept else model.w
        zero_count = int(np.sum(np.abs(w) < 1e-8))
        sparsity = float(zero_count / w.size)
        final_train_loss = model.history[-1] if model.history else None

        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": args.model,
            "solver": "N/A",
            "opt": args.opt,
            "lr": args.lr,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "loss": "logloss" if args.n_classes == 2 else "cross_entropy",
            "reg": args.reg,
            "alpha": args.alpha,
            "l1_ratio": args.l1_ratio,
            "fit_intercept": args.fit_intercept,
            "train_time_sec": train_time,
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2_score": None,
            "mape": None,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "zero_weights": zero_count,
            "sparsity": sparsity,
            "final_train_loss": final_train_loss,
        }
        log_run("./logs/runs.csv", row)

        print("=" * 100 + "\n")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        if args.reg in ("l1", "elastic"):
            print(f"Zero weights count: {zero_count}, sparsity: {sparsity}")

    print("\n" + "=" * 43 + " SANITY CHECK " + "=" * 43 + "\n")

    n_samples = X_train.shape[0]

    if args.model == "linear":
        if args.reg == "none":
            sk_model = SklearnLR(fit_intercept=args.fit_intercept)
        elif args.reg == "l2" or (
            args.reg == "elastic" and args.l1_ratio == 0
        ):
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
        sk_mse = mean_squared_error(y_test, sk_pred)

        print(f"My Model MSE:      {mse:.6f}")
        print(f"Sklearn Model MSE: {sk_mse:.6f}")

        diff = abs(mse - sk_mse)
        print(f"Difference:        {diff:.6f}")

    else:
        if args.reg == "l2":
            sk_model = SklearnLogReg(
                fit_intercept=args.fit_intercept,
                max_iter=args.steps,
                random_state=args.seed,
                C=1.0 / args.alpha,
                l1_ratio=0,
            )
        elif args.reg == "l1":
            sk_model = SklearnLogReg(
                fit_intercept=args.fit_intercept,
                max_iter=args.steps,
                random_state=args.seed,
                solver="saga",
                C=1.0 / args.alpha,
                l1_ratio=1,
            )
        elif args.reg == "elastic":
            sk_model = SklearnLogReg(
                fit_intercept=args.fit_intercept,
                max_iter=args.steps,
                random_state=args.seed,
                solver="saga",
                C=1.0 / args.alpha,
                l1_ratio=args.l1_ratio,
            )
        else:
            sk_model = SklearnLogReg(
                fit_intercept=args.fit_intercept,
                max_iter=args.steps,
                random_state=args.seed,
                C=np.inf,
            )

        sk_model.fit(X_train, y_train)
        sk_pred = sk_model.predict(X_test)
        sk_accuracy = accuracy_score(y_test, sk_pred)

        print(f"My Model Accuracy:      {accuracy:.6f}")
        print(f"Sklearn Model Accuracy: {sk_accuracy:.6f}")

        diff = abs(accuracy - sk_accuracy)
        print(f"Difference:             {diff:.6f}")


if __name__ == "__main__":
    main()
