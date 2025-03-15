import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict
from matplotlib import pyplot as plt
import seaborn as sns


from .data import (
    split_test_train_with_label,
    split_test_train_without_label,
)
from models.regression.base import (
    Model,
)



def train_model_for_feature(
    model_class: Model,
    X,
    y,
    seed=42,
):
    model = model_class()

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_test_train_with_label(
        X,
        y,
        test_size=0.2,
        random_state=seed,
    )

    model.fit(
        X_train,
        y_train,
    )

    return (
        model,
        X_test,
        y_test,
    )

def train_and_evaluate_model(
    target_column, 
    df=None,
    data_path=None, 
    feature_columns=None, 
    test_size=0.2, 
    random_state=42, 
    model_class:Model=None,
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
    data_path : str
        Path to the CSV data file
    target_column : str
        Name of the target variable column
    feature_columns : list or None, default=None
        List of feature columns to use. If None, uses all columns except target
    test_size : float, default=0.2
        Proportion of data to use for testing
    random_state : int, default=42
        Random seed for reproducibility
    model_class : class, default=None
        Model class to use (must have fit and predict methods)
    normalize_features : bool, default=True
        Whether to standardize features
    transform_target : callable or None, default=np.log
        Function to transform target variable
    transform_pred : callable or None, default=np.exp
        Function to transform predictions back to original scale
    fit_params : dict or None, default=None
        Parameters to pass to the model's fit method
    metrics : list or None, default=None
        List of metric names to calculate. Default: ['mse', 'r2']
    verbose : bool, default=True
        Whether to print results
        
    Returns:
    --------
    dict
        Dictionary with model, data splits, and evaluation metrics
    """
    if fit_params is None:
        fit_params = {'method': 'gradient_descent', 'learning_rate': 0.01, 'epochs': 1000}
    
    if metrics is None:
        metrics = ['mse', 'r2']
        
    if model_class is None:
       raise Exception("Es necesario especificar el modelo")
       
    df = pd.read_csv(data_path) if df is None else df
    
    # y_original = df[target_column].copy()
    if transform_target:
        df[target_column] = transform_target(df[target_column])
    
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    

    X_train, X_test, y_train, y_test = split_test_train_with_label(
        X, y, test_size=test_size, random_state=random_state
    )
    
    if normalize_features:

        X_train_mean = X_train.mean()
        X_train_std = X_train.std()
        
        X_train = (X_train - X_train_mean) / X_train_std
        
        X_test = (X_test - X_train_mean) / X_train_std
        
        normalization_params = {
            'mean': X_train_mean,
            'std': X_train_std
        }
    
    model:Model = model_class()
    model.fit(X_train, y_train, **fit_params)
    
    y_pred_test = model.predict(X_test)
    
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'feature_columns': feature_columns
    }
    
    if normalize_features:
        results['normalization_params'] = normalization_params
    
    metric_functions = {
        'mse': model.mse_score,
        'r2': model.r2_score,
        'mae': lambda x, y: np.mean(np.abs(y - model.predict(x)))
    }
    
    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](X_test, y_test)
    
    if verbose:
        print(f"\n=== Model Evaluation ({model_class.__name__}) ===")
        for metric in metrics:
            if metric in results:
                print(f"{metric.upper()}: {results[metric]:.6f}")
        
        try:
            model.print_coefficients()
        except:
            print("\nCoefficients:")
            for i, feat in enumerate(feature_columns):
                print(f"  {feat}: {model.coef_[i]:.6f}")
            print(f"  Intercept: {model.intercept_:.6f}")
            
    return results
