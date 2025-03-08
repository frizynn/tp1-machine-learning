import numpy as np
import pandas as pd


def split_test_train_with_label(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state=42,
    drop_target=True
    ):
    if drop_target:
        if (
            isinstance(
                y,
                pd.Series,
            )
            and y.name in X.columns
        ):
            X = X.drop(columns=[y.name])
        elif isinstance(
            y,
            pd.DataFrame,
        ):
            for col in y.columns:
                if col in X.columns:
                    X = X.drop(columns=[col])

    np.random.seed(random_state)
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)
    X = X.iloc[idx]
    y = y.iloc[idx]
    (
        X_train,
        X_test,
    ) = (
        X.iloc[:n_train],
        X.iloc[n_train:],
    )
    (
        y_train,
        y_test,
    ) = (
        y.iloc[:n_train],
        y.iloc[n_train:],
    )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


def split_test_train_without_label(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state=42,
):
    np.random.seed(random_state)
    n = len(df)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)
    df = df.iloc[idx]
    train, test = (
        df.iloc[:n_train],
        df.iloc[n_train:],
    )
    return (
        train,
        test,
    )
