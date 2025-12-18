import argparse
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from src.models import LinearRegression
from src.optimizers.gd import GD
from src.loss import MSE
from src.regularizers import L1, L2, Elastic_Net


def get_args():
    parser = argparse.ArgumentParser(description="Train Linear Regression model")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps")
    parser.add_argument(
        "--reg",
        type=str,
        choices=["none", "l1", "l2", "elastic"],
        default="none",
        help="Regularization type",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1, help="Regularization strength"
    )

    parser.add_argument("--n_samples", type=int, default=200, help="Number of samples")
    parser.add_argument("--n_features", type=int, default=50, help="Number of features")
    parser.add_argument("--noise", type=float, default=10.0, help="Noise level")

    return parser.parse_args()


def main():
    args = get_args()

    X, y = make_regression(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=5,
        noise=args.noise,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    regularizator = None
    if args.reg == "l1":
        regularizator = L1(alpha=args.alpha)
    elif args.reg == "l2":
        regularizator = L2(alpha=args.alpha)
    elif args.reg == "elastic":
        regularizator = Elastic_Net(alpha=args.alpha, l1_ratio=0.5)

    print(f"Training model with LR={args.lr}, Reg={args.reg}, Steps={args.steps}...")
    model = LinearRegression(
        fit_intercept=True,
        loss=MSE(),
        reg=regularizator,
        opt=GD(lr=args.lr),
        steps=args.steps,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    final_mse = MSE()(y_pred, y_test)

    print("-" * 100)
    print(f"Final MSE: {final_mse:.4f}")


if __name__ == "__main__":
    main()
