import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict
from matplotlib import pyplot as plt
import seaborn as sns


from models.regression.data import (
    split_test_train_with_label,
    split_test_train_without_label,
)
from models.regression.base import (
    Model,
)


def round_input(X, columnas):
    """
    Redondea a enteros las columnas especificadas de un DataFrame.
    """
    X_copy = X.copy()
    for col in columnas:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].round(0).astype(int)
    return X_copy


def get_nan_features(
    df: pd.DataFrame,
):
    nan_features = df.isnull().sum()
    nan_features = nan_features[nan_features > 0]
    nan_features = nan_features.to_dict()

    return nan_features


def split_by_nan_features(df, nan_features):

    exclusive_missing = []
    for (
        i,
        feat,
    ) in enumerate(nan_features):

        cond = df[feat].isnull()

        for (
            j,
            other,
        ) in enumerate(nan_features):
            if i != j:
                cond = cond & df[other].notnull()
        exclusive_missing.append(df[cond])

    cond_all_missing = df[nan_features].isnull().all(axis=1)
    all_missing = df[cond_all_missing]

    cond_all_present = df[nan_features].notnull().all(axis=1)
    all_present = df[cond_all_present]

    return (
        *exclusive_missing,
        all_missing,
        all_present,
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

def visualize_regression_results(
        y_true, 
        y_pred, 
        transform_func=None, 
        fig_size=(10, 5), 
        titles=None,
        save_path=None,
        show_figures=True
    ):
        """
        Creates comprehensive visualizations for regression model evaluation.
        
        Parameters:
        -----------
        y_true : array-like
            True target values
        y_pred : array-like
            Predicted target values
        transform_func : callable, optional
            Function to transform predictions and true values (e.g., np.exp)
        fig_size : tuple, default=(10, 5)
            Figure size for all plots
        titles : dict, optional
            Custom titles for plots. Keys: 'scatter', 'residuals', 'distribution', 'qq_plot'
        save_path : str, optional
            Directory path to save figures. If None, figures are not saved

        Returns:
        --------
        dict
            Dictionary containing the figures created
        """
        default_titles = {
            "scatter": "Precio Real vs Precio Predicho",
            "residuals": "Residuos vs Precio Predicho",
            "distribution": "Distribución de Residuos",
            "qq_plot": "Normal Q-Q Plot de Residuos"
        }
        
        if not titles:
            titles = default_titles
        else:
            for key in default_titles:
                if key not in titles:
                    titles[key] = default_titles[key]
        
        labels = {"actual": "Precio Real", "predicted": "Precio Predicho", "residuals": "Residuos"}
        
        if transform_func:
            y_true = transform_func(y_true)
            y_pred = transform_func(y_pred)
        
        figures = {}
        residuals = y_true - y_pred
        
        # actual vs predicted scatter
        fig_scatter = plt.figure(figsize=fig_size)
        sns.scatterplot(x=y_true, y=y_pred)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        plt.xlabel(labels["actual"])
        plt.ylabel(labels["predicted"])
        plt.title(titles["scatter"])
        if save_path:
            plt.savefig(f"{save_path}/scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.show() if show_figures else None
        figures["scatter"] = fig_scatter
        
        # residuos scatter
        fig_residuals = plt.figure(figsize=fig_size)
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(labels["predicted"])
        plt.ylabel(labels["residuals"])
        plt.title(titles["residuals"])
        if save_path:
            plt.savefig(f"{save_path}/residuals_plot.png", dpi=300, bbox_inches='tight')
        plt.show() if show_figures else None
        figures["residuals"] = fig_residuals
        
        # residuos distribution
        fig_dist = plt.figure(figsize=fig_size)
        sns.histplot(residuals, kde=True)
        plt.xlabel(labels["residuals"])
        plt.title(titles["distribution"])
        if save_path:
            plt.savefig(f"{save_path}/residuals_distribution.png", dpi=300, bbox_inches='tight')
        plt.show() if show_figures else None
        figures["distribution"] = fig_dist
        
        # Normal Q-Q plot of residuals
        fig_qq = plt.figure(figsize=fig_size)
        from scipy import stats
        
        # Calculate quantiles for the Q-Q plot
        (quantiles, ordered_values), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
        
        # Create the plot
        plt.scatter(quantiles, ordered_values)
        plt.plot(quantiles, slope * quantiles + intercept, 'r--')
        
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Cuantiles observados")
        plt.title(titles["qq_plot"])
        
        if save_path:
            plt.savefig(f"{save_path}/qq_plot.png", dpi=300, bbox_inches='tight')
        plt.show() if show_figures else None
        figures["qq_plot"] = fig_qq
        
        return figures

def train_and_evaluate_model(
    data_path, 
    target_column, 
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
    # Default parameters
    if fit_params is None:
        fit_params = {'method': 'gradient_descent', 'learning_rate': 0.01, 'epochs': 1000}
    
    if metrics is None:
        metrics = ['mse', 'r2']
        
    if model_class is None:
       raise Exception("Es necesario especificar el modelo")
       
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Apply transformation to target if specified
    y_original = df[target_column].copy()
    if transform_target:
        df[target_column] = transform_target(df[target_column])
    
    # Prepare features and target
    if feature_columns is None:
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    # Normalize features if requested
    if normalize_features:
        X = (X - X.mean()) / X.std()
    
    # Split data
    X_train, X_test, y_train, y_test = split_test_train_with_label(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train model
    model:Model = model_class()
    model.fit(X_train, y_train, **fit_params)
    
    # Predict
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'feature_columns': feature_columns
    }
    
    # Add requested metrics
    metric_functions = {
        'mse': model.mse_score,
        'r2': model.r2_score,
        'mae': lambda x, y: np.mean(np.abs(y - model.predict(x)))
    }
    
    for metric in metrics:
        if metric in metric_functions:
            results[metric] = metric_functions[metric](X_test, y_test)
    
    # Print results if requested
    if verbose:
        print(f"\n=== Model Evaluation ({model_class.__name__}) ===")
        for metric in metrics:
            if metric in results:
                print(f"{metric.upper()}: {results[metric]:.6f}")
        
        try:
            model.print_coefficients(metric="R2")
        except:
            print("\nCoefficients:")
            for i, feat in enumerate(feature_columns):
                print(f"  {feat}: {model.coef_[i]:.6f}")
            print(f"  Intercept: {model.intercept_:.6f}")
            
    return results