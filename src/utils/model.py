import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict
from matplotlib import pyplot as plt
import seaborn as sns


from .data import (
    split_test_train_with_label,
    load_and_prepare_data,
    normalize_data,
    print_model_evaluation,

)
from models.regression.base import (
    Model,
)



# def train_model_for_feature(
#     model_class: Model,
#     X,
#     y,
#     seed=42,
# ):
#     model = model_class()

#     (
#         X_train,
#         X_test,
#         y_train,
#         y_test,
#     ) = split_test_train_with_label(
#         X,
#         y,
#         test_size=0.2,
#         random_state=seed,
#     )

#     model.fit(
#         X_train,
#         y_train,
#     )

#     return (
#         model,
#         X_test,
#         y_test,
#     )


def evaluate_model(model, X_test, y_test, metrics=None):
    """
    Evaluate model performance using specified metrics.
    
    Parameters:
    -----------
    model : Model
        Trained model instance
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target values
    metrics : list, default=None
        List of metric names to calculate
        
    Returns:
    --------
    dict
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ['mse', 'r2']
        
    metric_functions = {
        'mse': model.mse_score,
        'r2': model.r2_score,
        'mae': lambda x, y: np.mean(np.abs(y - model.predict(x)))
    }
    
    results = {}
    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](X_test, y_test)
            
    return results


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
    transform_pred=None,
    fit_params=None,
    metrics=None,
    verbose=True
):
    """
    Load data, preprocess, train model, and evaluate performance.
    
    Parameters:
    -----------
    target_column : str
        Name of the target variable column
    df : pd.DataFrame, default=None
        DataFrame with data
    data_path : str, default=None
        Path to CSV file if df is not provided
    feature_columns : list, default=None
        List of feature columns to use. If None, uses all columns except target
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    model_class : class, default=None
        Model class to use (must have fit and predict methods)
    normalize_features : bool, default=True
        Whether to standardize features
    transform_target : callable, default=None
        Function to transform target variable
    transform_pred : callable, default=None
        Function to transform predictions back to original scale
    fit_params : dict, default=None
        Parameters to pass to the model's fit method
    metrics : list, default=None
        List of metric names to calculate
    verbose : bool, default=True
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary with model, data splits, and evaluation metrics
    """
    # Default parameters
    if fit_params is None:
        fit_params = {'method': 'gradient_descent', 'learning_rate': 0.01, 'epochs': 1000}
    
    if metrics is None:
        metrics = ['mse', 'r2']
        
    if model_class is None:
       raise Exception("Es necesario especificar el modelo")
    
    # Load and prepare data
    X, y, feature_columns = load_and_prepare_data(
        target_column, df, data_path, feature_columns, transform_target
    )
    
    # Split data
    X_train, X_test, y_train, y_test = split_test_train_with_label(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalize features if requested
    if normalize_features:
        X_train, X_test, normalization_params = normalize_data(X_train, X_test)
    
    # Train model
    model = model_class()
    model.fit(X_train, y_train, **fit_params)
    
    # Predict on test set
    y_pred_test = model.predict(X_test)
    
    # Evaluate model
    metrics_results = evaluate_model(model, X_test, y_test, metrics)
    
    # Print results if requested
    if verbose:
        print_model_evaluation(model, feature_columns, metrics_results)
    
    # Prepare results
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'feature_columns': feature_columns,
        **metrics_results
    }
    
    if normalize_features:
        results['normalization_params'] = normalization_params
    
    return results