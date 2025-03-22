from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models.regression.linear_regressor import LinearRegressor
from models.regression.base import Model


def visualize_regression_results( 
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

    def erfinv(y):
        # aproximación de la inversa de la función error (Winitzki)
        a = 0.147
        ln = np.log(1 - y**2)
        term1 = 2/(np.pi * a) + ln/2
        return np.sign(y) * np.sqrt(np.sqrt(term1**2 - ln/a) - term1)
    
    fig_qq = plt.figure(figsize=fig_size)
    
    ordered_values = np.sort(residuals)
    n = len(residuals)
    
    # calcular los porcentajes para cada punto (p = (i-0.5)/n)
    probs = (np.arange(1, n+1) - 0.5) / n
    
    # calcular los cuantiles teóricos de la distribución normal usando la función erfinv definida
    theoretical_quantiles = np.sqrt(2) * erfinv(2 * probs - 1)
    
    # ajustar una linea de referencia de regresión lineal entre los cuantiles teóricos y los valores observados
    slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, 1)
    
    # correlacion entre los cuantiles teoricos y observados
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

def plot_metric_vs_lambda(ax, lambdas, metric_scores, metric_name="MSE",marker='o'):
    """
    Grafica una métrica (MSE o R²) en función de λ.
    
    Parámetros:
        - ax: Objeto de ejes de Matplotlib donde se dibujará el gráfico.
        - lambdas: Lista o array de valores de λ.
        - metric_scores: Lista o array de valores de la métrica.
        - metric_name: Nombre de la métrica ("MSE" o "R2"). Por defecto "MSE".
    
    Retorna:
        - optimal_lambda: Valor de λ que optimiza la métrica (minimiza MSE o maximiza R²).
        - optimal_metric: Valor óptimo de la métrica obtenido.
    """
    # Graficar la línea de la métrica
    sns.lineplot(x=lambdas, y=metric_scores, ax=ax, color='royalblue', label=metric_name,
                marker=marker)

    if metric_name.upper() == "R2":
        # Para R², buscamos el máximo
        optimal_idx = np.argmax(metric_scores)
        optimal_metric = metric_scores[optimal_idx]
        optimal_lambda = lambdas[optimal_idx]
        optimal_type = "Max"
    else:
        # Para MSE u otras métricas de error, buscamos el mínimo
        optimal_idx = np.argmin(metric_scores)
        optimal_metric = metric_scores[optimal_idx]
        optimal_lambda = lambdas[optimal_idx]
        optimal_type = "Min"
    
    # Marcar el punto óptimo con un punto y una línea vertical discontinua
    ax.scatter([optimal_lambda], [optimal_metric], color='red', s=100, zorder=5,
                label=f'{optimal_type} {metric_name}: {optimal_metric:.4f} (λ = {optimal_lambda:.4f})')
    ax.axvline(x=optimal_lambda, color='red', linestyle='--', alpha=0.7)
    
    # Agregar texto con valores óptimos
    ax.text(0.02, 0.98, f'λ óptimo: {optimal_lambda:.4f}\n{metric_name} {optimal_type.lower()}: {optimal_metric:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8),
            verticalalignment='top')
    
    ax.set_xlabel('Regularization strength (λ)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs λ')
    ax.grid(True)
    
    return optimal_lambda, optimal_metric

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
    
    min_lambda, min_mse = plot_metric_vs_lambda(ax1, lambdas, mse_scores, metric_name="MSE",marker=None)
    max_lambda, max_r2 = plot_metric_vs_lambda(ax2, lambdas, r2_scores, metric_name="R2",marker=None)
    
    plt.tight_layout()
    plt.suptitle('Métricas de rendimiento vs Regularization Strength', y=1.05, fontsize=16)
    plt.show()
    
    return min_lambda, min_mse, max_lambda, max_r2

def plot_cv_results(lambdas, cv_mse_scores, optimal_lambda, min_cv_mse, title=None, ax=None):
    """
    Grafica la variación del MSE promedio (de validación cruzada) en función de λ utilizando seaborn.
    
    Parámetros:
      - lambdas: Secuencia de valores de λ.
      - cv_mse_scores: MSE promedio correspondiente a cada λ.
      - optimal_lambda: Valor de λ que minimiza el MSE.
      - min_cv_mse: MSE mínimo obtenido.
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
    data = pd.DataFrame({'lambda': lambdas, 'MSE': cv_mse_scores})
    sns.lineplot(data=data, x='lambda', y='MSE', ax=ax, label='MSE promedio')
    
    # Marcar el lambda óptimo con un punto y una línea vertical discontinua incluida en la leyenda
    ax.scatter([optimal_lambda], [min_cv_mse], color='red', s=100, zorder=5,
               label=f'Min MSE: {min_cv_mse:.4f} (λ = {optimal_lambda:.4f})')
    ax.axvline(x=optimal_lambda, color='red', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Intensidad de regularización (λ)', fontsize=12)
    ax.set_ylabel('MSE - Validación Cruzada', fontsize=12)
    if title is None:
        title = 'Validación Cruzada: MSE vs λ'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax