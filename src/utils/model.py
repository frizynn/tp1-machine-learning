import numpy as np

from .data import (
    evaluate_model,
    load_and_prepare_data,
    mse_score,
    normalize_data,
    r2_score,
    split_test_train,
)


def prepare_and_evaluate_test_data(
    df_test,
    model,
    normalization_params=None,
    target_column='price',
    feature_columns=None,
    transform_target_func=None,
    inv_transform_pred=None,
    metrics=None,
    print_metrics=False,
    round_digits=4,
):
    """
    Prepares test data by normalizing features and transforming the target variable,
    then evaluates a model.

    Parameters:
    -----------
    df_test : pd.DataFrame
        test data with features and target variable
    model : object
        trained model to evaluate
    normalization_params : dict, default=None
        dictionary with 'mean' and 'std' normalization parameters from training
        if None, normalization is not applied
    target_column : str, default='price'
        name of the target column in df_test
    feature_columns : list, default=None
        list of feature columns to use. if None, uses all columns except the target
    transform_target_func : callable, default=None
        function to transform the target variable (e.g., np.log)
        if None, no transformation is applied
    inv_transform_pred : callable, default=None
        function to transform predictions back to original scale (e.g., np.exp)
    metrics : list, default=None
        list of metric functions to calculate
    print_metrics : bool, default=False
        indicates whether to print formatted metric results
    round_digits : int, default=4
        number of decimal places to round metrics when printing

    Returns:
    --------
    tuple
        (metrics_dict, X_test, y_test, y_test_transformed)
        - metrics_dict: dictionary with evaluation metrics
        - X_test: processed feature matrix
        - y_test: original target variable values
        - y_test_transformed: transformed target variable values (if transformation was applied)
    """
    if feature_columns is None:
        feature_columns = [col for col in df_test.columns if col != target_column]

    X_test = (
        df_test[feature_columns].copy()
        if feature_columns
        else df_test.drop(target_column, axis=1)
    )
    y_test = df_test[target_column].copy()

    if normalization_params is not None:
        for col in X_test.columns:
            if (
                col in normalization_params['mean']
                and col in normalization_params['std']
            ):
                X_test[col] = (
                    X_test[col] - normalization_params['mean'][col]
                ) / normalization_params['std'][col]

    if transform_target_func is not None:
        y_test_transformed = transform_target_func(y_test)
    else:
        y_test_transformed = y_test

    metrics_dict = evaluate_model(
        model, X_test, y_test_transformed, metrics, inv_transform_pred
    )

    if print_metrics:
        for metric_name, value in metrics_dict.items():
            print(f"{metric_name.upper()}: {value:.{round_digits}f}")

    return metrics_dict, X_test, y_test, y_test_transformed


def train_and_evaluate_model(
    target_column,
    df=None,
    data_path=None,
    feature_columns=None,
    test_size=0.2,
    random_state=42,
    model_class=None,
    normalize_features=True,
    transform_target=None,
    inv_transform_pred=None,
    fit_params=None,
    metrics=None,
    verbose=True,
):
    """
    Loads data, preprocesses, trains model and evaluates performance.

    Parameters:
    -----------
    target_column : str
        Name of the target column
    df : pd.DataFrame, default=None
        DataFrame with the data
    data_path : str, default=None
        Path to CSV file if df is not provided
    feature_columns : list, default=None
        List of feature columns to use. If None, uses all columns except the target
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    model_class : class, default=None
        Model class to use (must have fit and predict methods)
    normalize_features : bool, default=True
        Indicates whether to standardize features
    transform_target : callable, default=None
        Function to transform the target variable
    inv_transform_pred : callable, default=None
        Function to transform predictions back to original scale
    fit_params : dict, default=None
        Parameters to pass to the model's fit method
    metrics : list, default=None
        List of metric functions to calculate
    verbose : bool, default=True
        Indicates whether to print results

    Returns:
    --------
    dict
        Dictionary with model, data splits and evaluation metrics
    """

    # default parameters
    if fit_params is None:
        fit_params = {
            'method': 'gradient_descent',
            'learning_rate': 0.01,
            'epochs': 1000,
        }

    if metrics is None:
        metrics = [mse_score, r2_score]

    if model_class is None:
        raise Exception("Model specification is required")

    if transform_target and not inv_transform_pred:
        raise Exception("inv_transform_pred function must be specified")

    # load and prepare data
    X, y, feature_columns = load_and_prepare_data(
        target_column, df, data_path, feature_columns, transform_target
    )

    # split data
    X_train, X_test, y_train, y_test = split_test_train(
        X, y, test_size=test_size, random_state=random_state
    )

    # normalize features if requested
    if normalize_features:
        X_train, X_test, normalization_params = normalize_data(X_train, X_test)

    # train model
    model = model_class()
    model.fit(X_train, y_train, **fit_params)

    # predict on test set
    y_pred_test = model.predict(X_test)

    # evaluate model
    metrics_results = evaluate_model(
        model, X_test, y_test, metrics, inv_transform_pred, y_pred_test
    )

    if verbose:
        model.print_coefficients()

    # prepare results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'feature_columns': feature_columns,
        **metrics_results,
    }

    if normalize_features:
        results['normalization_params'] = normalization_params

    return results


def get_weights_and_metrics(
    X,
    y,
    lambdas,
    model_class,
    test_size=0.2,
    random_state=42,
    normalize=True,
    regularization='l2',
    method='pseudo_inverse',
    inv_transform_pred=None,
    metrics=None,
):
    """
    Gets model weights and performance metrics for different lambda values.

    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target variable
    lambdas : array-like
        Regularization parameter values to test
    model_class : class
        Model class to use
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    normalize : bool, default=True
        Indicates whether to normalize features
    regularization : str, default='l2'
        Type of regularization to use (l1 or l2)
    method : str, default='pseudo_inverse'
        Method to use for solving the linear system
    inv_transform_pred : callable, default=None
        Function to transform predictions back to original scale
    metrics : list, default=None
        List of metric functions to calculate

    Returns:
    --------
    tuple
        (weights, mse_scores, r2_scores) arrays
    """
    if metrics is None:
        metrics = [mse_score, r2_score]

    X_train, X_test, y_train, y_test = split_test_train(
        X, y, test_size=test_size, random_state=random_state, normalize=normalize
    )

    weights = []
    metric_scores = {metric.__name__.lower(): [] for metric in metrics}

    for lambda_ in lambdas:
        model = model_class()
        model.fit(
            X_train,
            y_train,
            method=method,
            alpha=lambda_,
            regularization=regularization,
        )

        coefs = model.get_weights()
        weights.append(coefs)

        score = evaluate_model(model, X_test, y_test, metrics, inv_transform_pred)

        for key, value in score.items():
            metric_scores[key].append(value)

    weights = np.array(weights)

    mse_scores = metric_scores.get('mse_score', metric_scores.get('mse', []))
    r2_scores = metric_scores.get('r2_score', metric_scores.get('r2', []))

    return weights, mse_scores, r2_scores
