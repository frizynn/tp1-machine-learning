from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models.regression.linear_regressor import LinearRegressor
from models.regression.base import Model


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

def compare_feature_impact(model_results_dict, property_dfs, feature_name='has_pool', 
                        currency='$', 
                         ):
    """
    Compare the impact of a specific feature across different models.
    
    Parameters
    ----------
    model_results_dict : dict
        Dictionary mapping model names to model results (from train_and_evaluate_model),
        or a single model result dictionary (will be treated as a single model)
    property_dfs : dict
        Dictionary mapping model names to corresponding DataFrames,
        or a single DataFrame (will be treated as a single model)
    feature_name : str, default='has_pool'
        Name of the feature to analyze impact
    plot : bool, default=False
        Whether to generate and display visualization (default changed to False)
    y_lim : tuple or None, default=None
        Custom y-axis limits for absolute impact (min, max)
    x_lim : tuple or None, default=None
        Custom x-axis limits for property price range (min, max)
    currency : str, default='$'
        Currency symbol to use in labels and titles
    custom_title : str or None, default=None
        Custom plot title, if None a default is generated
    style : str, default='whitegrid'
        Seaborn plotting style to use
    fixed_x_ticks : array-like or None, default=None
        Custom tick positions for x-axis
    fixed_y1_ticks : array-like or None, default=None
        Custom tick positions for primary y-axis (absolute impact)
    fixed_y2_ticks : array-like or None, default=None
        Custom tick positions for secondary y-axis (percentage impact)
    
    Returns
    -------
    dict
        Dictionary containing analysis results for each model
    """

    if not isinstance(model_results_dict, dict) or (isinstance(model_results_dict, dict) and 'model' in model_results_dict):
        model_results_dict = {"Modelo": model_results_dict}
        property_dfs = {"Modelo": property_dfs}
    
    if not isinstance(property_dfs, dict):
        raise ValueError("property_dfs must be a dictionary")
    
    if set(model_results_dict.keys()) != set(property_dfs.keys()):
        raise ValueError("model_results_dict and property_dfs must have the same keys")
    

    impacts = {}
    for model_name, model_results in model_results_dict.items():

        coef_dict = None
        if hasattr(model_results["model"], 'get_coef_dict') and callable(model_results["model"].get_coef_dict):
            coef_dict = model_results["model"].get_coef_dict()
        elif hasattr(model_results["model"], 'coef_dict') and model_results["model"].coef_dict:
            coef_dict = model_results["model"].coef_dict
        
        if not coef_dict or feature_name not in coef_dict:
            print(f"Warning: Feature '{feature_name}' not found in model '{model_name}'. Skipping.")
            continue
            
        df = property_dfs[model_name]
        feature_impact = coef_dict[feature_name]
        avg_property_price = df["price"].mean()
        percentage_impact = (feature_impact / avg_property_price) * 100
        
        display_name = feature_name.replace('_', ' ').replace('has ', '').title()
        
        
        
        impacts[model_name] = {
            "feature_name": feature_name,
            "display_name": display_name,
            "absolute_impact": feature_impact,
            "average_price": avg_property_price,
            "percentage_impact": percentage_impact
        }
        
    return impacts

def plot_weights_vs_lambda(lambdas, weights, feature_names, custom_titles=None):
    """
    Grafica los valores de los pesos en función de la regularización λ usando seaborn.
    
    Parámetros:
    - lambdas: Lista o array de valores de λ.
    - weights: Matriz donde cada columna representa los pesos de una característica en función de λ.
    - feature_names: Lista de nombres de las características.
    - custom_titles: Diccionario con títulos personalizados. Claves: 'title', 'xlabel', 'ylabel'
    """
    # Default titles
    titles = {
        "title": "Ridge Regression: Weight Values vs Regularization Strength",
        "xlabel": "Regularization strength (λ)", 
        "ylabel": "Weight Value"
    }
    
    # Update with custom titles if provided
    if custom_titles:
        titles.update(custom_titles)

    plt.figure(figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # Crear un DataFrame para usar con seaborn
    data = pd.DataFrame(weights, columns=feature_names[:weights.shape[1]])
    data['lambda'] = lambdas
    
    # Convertir a formato long para seaborn
    data_long = pd.melt(data, id_vars=['lambda'], var_name='feature', value_name='weight')
    
    # Graficar usando seaborn
    ax = sns.lineplot(x='lambda', y='weight', hue='feature', data=data_long)
    
    plt.xlabel(titles["xlabel"], fontsize=12)
    plt.ylabel(titles["ylabel"], fontsize=12)
    plt.title(titles["title"], fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Mover la leyenda fuera del gráfico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
    
    plt.tight_layout()
    plt.show()

def plot_mse_vs_lambda(ax, lambdas, mse_scores):
    """
    Grafica el Error Cuadrático Medio (MSE) en función de λ.
    
    Parámetros:
        - ax: Objeto de ejes de Matplotlib donde se dibujará el gráfico.
        - lambdas: Lista o array de valores de λ.
        - mse_scores: Lista o array de valores de MSE.
    
    Retorna:
        - min_lambda: Valor de λ que minimiza el MSE.
        - min_mse: MSE mínimo obtenido.
    """
    # Graficar la línea del MSE
    sns.lineplot(x=lambdas, y=mse_scores, ax=ax, color='blue', label='MSE')
    
    # Calcular el mínimo
    min_mse_idx = np.argmin(mse_scores)
    min_mse = mse_scores[min_mse_idx]
    min_lambda = lambdas[min_mse_idx]
    
    # Marcar el mínimo con un punto y una línea vertical discontinua
    ax.scatter([min_lambda], [min_mse], color='red', s=100, zorder=5,
                label=f'Min MSE: {min_mse:.4f} (λ = {min_lambda:.4f})')
    ax.axvline(x=min_lambda, color='red', linestyle='--', alpha=0.7)
    
    # Agregar texto con valores óptimos
    ax.text(0.02, 0.98, f'λ óptimo: {min_lambda:.4f}\nMSE mínimo: {min_mse:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlabel('Regularization strength (λ)')
    ax.set_ylabel('Error cuadrático medio')
    ax.set_title('MSE vs λ')
    ax.grid(True)
    
    return min_lambda, min_mse

def plot_r2_vs_lambda(ax, lambdas, r2_scores):
    """
    Grafica el coeficiente de determinación R² en función de λ.
    
    Parámetros:
        - ax: Objeto de ejes de Matplotlib donde se dibujará el gráfico.
        - lambdas: Lista o array de valores de λ.
        - r2_scores: Lista o array de valores de R².
    
    Retorna:
        - max_lambda: Valor de λ que maximiza R².
        - max_r2: Valor máximo de R² obtenido.
    """
    sns.lineplot(x=lambdas, y=r2_scores, ax=ax, color='green', label='R²')
    
    # Calcular el máximo
    max_r2_idx = np.argmax(r2_scores)
    max_r2 = r2_scores[max_r2_idx]
    max_lambda = lambdas[max_r2_idx]
    
    # Marcar el máximo con un punto y una línea vertical discontinua
    ax.scatter([max_lambda], [max_r2], color='red', s=100, zorder=5,
                label=f'Max R²: {max_r2:.4f} (λ = {max_lambda:.4f})')
    ax.axvline(x=max_lambda, color='red', linestyle='--', alpha=0.7)
    
    # Agregar texto con valores óptimos
    ax.text(0.02, 0.98, f'λ óptimo: {max_lambda:.4f}\nR² máximo: {max_r2:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlabel('Regularization strength (λ)')
    ax.set_ylabel('Coeficiente de determinación (R²)')
    ax.set_title('R² vs λ')
    ax.grid(True)
    
    return max_lambda, max_r2

def plot_performance_metrics(lambdas, mse_scores, r2_scores):
    """
    Crea subgráficos con las métricas MSE y R² en función de λ.
    
    Parámetros:
      - lambdas: Lista o array de valores de λ.
      - mse_scores: Lista o array de valores de MSE.
      - r2_scores: Lista o array de valores de R².
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    min_lambda, min_mse = plot_mse_vs_lambda(ax1, lambdas, mse_scores)
    max_lambda, max_r2 = plot_r2_vs_lambda(ax2, lambdas, r2_scores)
    
    plt.tight_layout()
    plt.suptitle('Métricas de rendimiento vs Regularization Strength', y=1.05, fontsize=16)
    plt.show()
    
    return min_lambda, min_mse, max_lambda, max_r2

def plot_cv_results(lambdas, cv_mse_scores, optimal_lambda, min_cv_mse, title=None, ax=None):
    """
    Grafica la variación del ECM promedio (de validación cruzada) en función de λ utilizando seaborn.
    
    Parámetros:
      - lambdas: Secuencia de valores de λ.
      - cv_mse_scores: ECM promedio correspondiente a cada λ.
      - optimal_lambda: Valor de λ que minimiza el ECM.
      - min_cv_mse: ECM mínimo obtenido.
      - title (str, opcional): Título del gráfico.
      - ax (matplotlib.axes.Axes, opcional): Eje sobre el cual dibujar.
    
    Retorna:
      - ax: Objeto matplotlib.axes.Axes con el gráfico.
    """
    if ax is None:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    sns.set_style("whitegrid")
    
    # Crear DataFrame para graficar usando seaborn
    data = pd.DataFrame({'lambda': lambdas, 'ECM': cv_mse_scores})
    sns.lineplot(data=data, x='lambda', y='ECM', ax=ax, label='ECM promedio')
    
    # Marcar el lambda óptimo con un punto y una línea vertical discontinua incluida en la leyenda
    ax.scatter([optimal_lambda], [min_cv_mse], color='red', s=100, zorder=5,
               label=f'Min ECM: {min_cv_mse:.4f} (λ = {optimal_lambda:.4f})')
    ax.axvline(x=optimal_lambda, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Intensidad de regularización (λ)', fontsize=12)
    ax.set_ylabel('Error Cuadrático Medio (ECM) - Validación Cruzada', fontsize=12)
    if title is None:
        title = 'Validación Cruzada: ECM vs λ'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax