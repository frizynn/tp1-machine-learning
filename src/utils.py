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
            Custom titles for plots. Keys: 'scatter', 'residuals', 'distribution'
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
            "distribution": "Distribución de Residuos"
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
        
        # residuos distrbution
        fig_dist = plt.figure(figsize=fig_size)
        sns.histplot(residuals, kde=True)
        plt.xlabel(labels["residuals"])
        plt.title(titles["distribution"])
        if save_path:
            plt.savefig(f"{save_path}/residuals_distribution.png", dpi=300, bbox_inches='tight')
        plt.show() if show_figures else None
        figures["distribution"] = fig_dist
        
        return figures
