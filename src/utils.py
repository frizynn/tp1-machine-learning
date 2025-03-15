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

def visualize_regression_results( #TODO: implementar parametro de que figuras ver
        y_true, 
        y_pred, 
        transform_func=None, 
        fig_size=(10, 5), 
        titles=None,
        save_path=None,
        show_figures=True,
        fit_degree=1
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
    show_figures : bool, default=True
        Whether to display the figures
    fit_degree : int, default=1
        Degree of polynomial to fit (1=linear, 2=quadratic, etc.)

    Returns:
    --------
    dict
        Dictionary containing the figures created
    """
    def _initialize_plot_titles(titles):
        """Initialize and validate plot titles."""
        default_titles = {
            "scatter": "Precio Real vs Precio Predicho",
            "residuals": "Residuos vs Precio Predicho",
            "distribution": "Distribución de Residuos",
            "qq_plot": "Normal Q-Q Plot de Residuos"
        }
        
        if not titles:
            return default_titles
        else:
            for key in default_titles:
                if key not in titles:
                    titles[key] = default_titles[key]
            return titles

    titles = _initialize_plot_titles(titles)
    labels = {"actual": "Precio Real", "predicted": "Precio Predicho", "residuals": "Residuos"}
    
    if transform_func:
        y_true = transform_func(y_true)
        y_pred = transform_func(y_pred)
    
    residuals = y_true - y_pred
    
    figures = {}
    figures["scatter"] = _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, 
                                            save_path, show_figures, fit_degree)
    figures["residuals"] = _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, 
                                                save_path, show_figures)
    figures["distribution"] = _create_distribution_plot(residuals, labels, titles, fig_size, 
                                                      save_path, show_figures)
    figures["qq_plot"] = _create_qq_plot(residuals, titles, fig_size,  save_path, show_figures)
    
    return figures




def _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, save_path, show_figures, fit_degree):
    """Create scatter plot of actual vs predicted values with polynomial fit."""
    fig_scatter = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, label="Datos")
    
    x_data = np.array(y_true)
    y_data = np.array(y_pred)
    

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', 
             label="Predicción perfecta")
    
    # agregar polinomio si fit degree > 1
    if fit_degree > 1:
        coeffs = np.polyfit(x_data, y_data, fit_degree)
        p = np.poly1d(coeffs)
        
        x_curve = np.linspace(min_val, max_val, 100)
        y_curve = p(x_curve)
        
        plt.plot(x_curve, y_curve, 'r-', 
                 label=f"Ajuste polinómico (grado={fit_degree})")
    
    plt.xlabel(labels["actual"])
    plt.ylabel(labels["predicted"])
    plt.title(titles["scatter"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if save_path:
        plt.savefig(f"{save_path}/scatter_plot.png", dpi=300, bbox_inches='tight')
    plt.show() if show_figures else plt.close()
    
    return fig_scatter


def _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, save_path, show_figures):
    """Create scatter plot of residuals vs predicted values."""
    fig_residuals = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, label="Residuos")
    
    plt.axhline(y=0, color='r', linestyle='--', label="Residuo cero")
    
    plt.xlabel(labels["predicted"])
    plt.ylabel(labels["residuals"])
    plt.title(titles["residuals"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if save_path:
        plt.savefig(f"{save_path}/residuals_plot.png", dpi=300, bbox_inches='tight')
    plt.show() if show_figures else plt.close()
    
    return fig_residuals

def _create_distribution_plot(residuals, labels, titles, fig_size, save_path, show_figures):
    """Create histogram with KDE of residuals using only numpy, seaborn and matplotlib."""
    fig_dist = plt.figure(figsize=fig_size)
    
    sns.histplot(residuals, kde=True, stat="density", label="Distribución")
    
    plt.xlabel(labels["residuals"])
    plt.title(titles["distribution"])
    plt.grid(True, alpha=0.3, linestyle='--')
    
    x = np.linspace(np.min(residuals), np.max(residuals), 100)
    mean = np.mean(residuals)
    std = np.std(residuals)
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    plt.plot(x, normal_pdf, 'r-', label="Distribución normal")
    plt.legend()
    
    if save_path:
        plt.savefig(f"{save_path}/residuals_distribution.png", dpi=300, bbox_inches='tight')
    plt.show() if show_figures else plt.close()
    
    return fig_dist



def _create_qq_plot(residuals, titles, fig_size, save_path, show_figures):
    """Crea un gráfico Q-Q de los residuos contra una distribución normal utilizando numpy y matplotlib."""

    def erfinv(y):
        # Aproximación de la inversa de la función error (Winitzki)
        a = 0.147
        ln = np.log(1 - y**2)
        term1 = 2/(np.pi * a) + ln/2
        return np.sign(y) * np.sqrt(np.sqrt(term1**2 - ln/a) - term1)
    
    fig_qq = plt.figure(figsize=fig_size)
    
    # Ordenar los residuos
    ordered_values = np.sort(residuals)
    n = len(residuals)
    
    # Calcular los porcentajes para cada punto (p = (i-0.5)/n)
    probs = (np.arange(1, n+1) - 0.5) / n
    
    # Calcular los cuantiles teóricos de la distribución normal usando la función erfinv definida
    theoretical_quantiles = np.sqrt(2) * erfinv(2 * probs - 1)
    
    # Ajustar una línea de referencia: regresión lineal entre los cuantiles teóricos y los valores observados
    slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, 1)
    
    # Calcular la correlación entre los cuantiles teóricos y observados para mostrarla en la leyenda
    r = np.corrcoef(theoretical_quantiles, ordered_values)[0,1]
    
    plt.scatter(theoretical_quantiles, ordered_values, alpha=0.7, label="Cuantiles")
    plt.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, 'r--', 
             label=f"Línea de referencia (r={r:.3f})")
    
    plt.xlabel("Cuantiles teóricos")
    plt.ylabel("Cuantiles observados")
    plt.title(titles["qq_plot"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if save_path:
        plt.savefig(f"{save_path}/qq_plot.png", dpi=300, bbox_inches='tight')
    plt.show() if show_figures else plt.close()
    
    return fig_qq


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
    
    y_original = df[target_column].copy()
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
            model.print_coefficients(metric="R2")
        except:
            print("\nCoefficients:")
            for i, feat in enumerate(feature_columns):
                print(f"  {feat}: {model.coef_[i]:.6f}")
            print(f"  Intercept: {model.intercept_:.6f}")
            
    return results



def analyze_pool_value_impact(model_results, property_df, feature_name='has_pool', plot=True):
    """
    Analyze the impact of a specific feature (e.g., pool) on property value.
    
    Parameters
    ----------
    model_results : dict
        Dictionary containing model and results from train_and_evaluate_model
    property_df : pandas.DataFrame
        DataFrame containing property data used in the model
    feature_name : str, default='has_pool'
        Name of the feature to analyze impact
    plot : bool, default=True
        Whether to generate and display visualization
        
    Returns
    -------
    dict
        Dictionary containing analysis results (impact values and percentages)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Validate inputs
    if feature_name not in model_results["model"].get_coef_dict():
        raise ValueError(f"Feature '{feature_name}' not found in model coefficients.")
    
    # Extract feature coefficient and calculate impacts
    feature_impact = model_results["model"].get_coef_dict()[feature_name]
    avg_property_price = property_df["price"].mean()
    percentage_impact = (feature_impact / avg_property_price) * 100
    
    # Print results
    print(f"\nImpact of adding {feature_name.replace('_', ' ')} to a property:")
    print(f"- Absolute value added: {feature_impact:.2f} monetary units")
    print(f"- Average property price: {avg_property_price:.2f} monetary units")
    print(f"- Percentage increase on average: {percentage_impact:.2f}%")
    
    # Property-specific impact calculation
    if property_df["is_house"].nunique() == 1:
        property_type = "house" if property_df["is_house"].iloc[0] == 1 else "department"
        print(f"\nNote: This model was trained on {property_type}s only.")
    
    # Generate visualization if requested
    if plot:
        _visualize_impact_vs_price(feature_impact, property_df)
    
    # Return results as a dictionary for potential further use
    return {
        "feature_name": feature_name,
        "absolute_impact": feature_impact,
        "average_price": avg_property_price,
        "percentage_impact": percentage_impact
    }


def _visualize_impact_vs_price(impact_value, property_df, num_points=10):
    """
    Visualize how feature impact varies across different property prices.
    
    Parameters
    ----------
    impact_value : float
        The impact value of the feature
    property_df : pandas.DataFrame
        DataFrame containing property data
    num_points : int, default=10
        Number of points to use in the visualization
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate price points for visualization
    min_price = max(10, property_df["price"].min())  # Avoid division by zero
    max_price = property_df["price"].max()
    property_prices = np.linspace(min_price, max_price, num_points)
    
    # Calculate absolute and percentage impacts
    absolute_values = [impact_value] * num_points
    percent_values = [impact_value / price * 100 for price in property_prices]
    
    # Create figure with two y-axes for different scales
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot absolute impact
    color = 'tab:blue'
    ax1.set_xlabel('Property Price')
    ax1.set_ylabel('Absolute Value Added', color=color)
    ax1.plot(property_prices, absolute_values, color=color, label="Absolute Value")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=impact_value, color=color, linestyle='--', alpha=0.5)
    
    # Create second y-axis for percentage impact
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Percentage Value Added (%)', color=color)
    ax2.plot(property_prices, percent_values, color=color, label="Percentage Value")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add title and grid
    plt.title('Impact of Feature vs. Property Price')
    ax1.grid(True, alpha=0.3)
    
    # Add legend with both lines
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()