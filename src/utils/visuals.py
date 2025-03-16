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
                         plot=True, y_lim=None, x_lim=None, currency='$', 
                         custom_title=None, style='whitegrid', fixed_x_ticks=None, 
                         fixed_y1_ticks=None, fixed_y2_ticks=None):
    """
    Compare the impact of a specific feature across different models.
    
    Parameters
    ----------
    model_results_dict : dict
        Dictionary mapping model names to model results (from train_and_evaluate_model)
    property_dfs : dict
        Dictionary mapping model names to corresponding DataFrames
    feature_name : str, default='has_pool'
        Name of the feature to analyze impact
    plot : bool, default=True
        Whether to generate and display visualization
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
    # Validate inputs
    if not isinstance(model_results_dict, dict) or not isinstance(property_dfs, dict):
        raise ValueError("model_results_dict and property_dfs must be dictionaries")
    
    if set(model_results_dict.keys()) != set(property_dfs.keys()):
        raise ValueError("model_results_dict and property_dfs must have the same keys")
    
    # Calculate impacts and store results
    impacts = {}
    for model_name, model_results in model_results_dict.items():
        if feature_name not in model_results["model"].get_coef_dict():
            print(f"Warning: Feature '{feature_name}' not found in model '{model_name}'. Skipping.")
            continue
            
        df = property_dfs[model_name]
        feature_impact = model_results["model"].get_coef_dict()[feature_name]
        avg_property_price = df["price"].mean()
        percentage_impact = (feature_impact / avg_property_price) * 100
        
        display_name = feature_name.replace('_', ' ').replace('has ', '').title()
        
        print(f"\nImpact of adding {display_name} for {model_name}:")
        print(f"- Absolute value added: {currency}{feature_impact:.2f}")
        print(f"- Average property price: {currency}{avg_property_price:.2f}")
        print(f"- Percentage increase on average: {percentage_impact:.2f}%")
        
        impacts[model_name] = {
            "feature_name": feature_name,
            "display_name": display_name,
            "absolute_impact": feature_impact,
            "average_price": avg_property_price,
            "percentage_impact": percentage_impact
        }
    
    if plot and impacts:
        plot_result = _compare_impact_vs_price(
            impacts, 
            property_dfs,
            feature_name=feature_name.replace('_', ' ').replace('has ', '').title(), 
            y_lim=y_lim,
            x_lim=x_lim,
            currency=currency,
            custom_title=custom_title,
            style=style,
            fixed_x_ticks=fixed_x_ticks,
            fixed_y1_ticks=fixed_y1_ticks,
            fixed_y2_ticks=fixed_y2_ticks
        )
    
    return impacts
def _compare_impact_vs_price(impacts, property_dfs, feature_name, y_lim=None, x_lim=None, 
                            num_points=10, currency='$', custom_title=None, style='whitegrid', 
                            fixed_x_ticks=None, fixed_y1_ticks=None, fixed_y2_ticks=None):
    """
    Visualize how impact of a feature varies across different property price ranges for multiple models.
    
    Parameters
    ----------
    impacts : dict
        Dictionary with impact analysis results for each model
    property_dfs : dict
        Dictionary with property DataFrames for each model
    feature_name : str
        Name of the feature for titles and labels
    y_lim : tuple or None, default=None
        Custom y-axis limits for absolute impact (min, max)
    x_lim : tuple or None, default=None
        Custom x-axis limits for price range (min, max)
    num_points : int, default=10
        Number of points to use in the visualization
    currency : str, default='$'
        Currency symbol to use in labels and titles
    custom_title : str or None, default=None
        Custom plot title; if None a default is generated
    style : str, default='whitegrid'
        Seaborn style to use
    fixed_x_ticks : array-like or None, default=None
        Custom tick positions for x-axis
    fixed_y1_ticks : array-like or None, default=None
        Custom tick positions for primary y-axis (absolute impact)
    fixed_y2_ticks : array-like or None, default=None
        Custom tick positions for secondary y-axis (percentage impact)
    
    Returns
    -------
    dict
        Dictionary containing the plot objects
    """
    # Apply Seaborn style
    sns.set_style(style)
    
    # Create figure with two subplots (one above the other)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Determine price range for x-axis
    if x_lim:
        min_price, max_price = x_lim
    else:
        min_prices = [max(10, df["price"].min()) for df in property_dfs.values()]
        max_prices = [df["price"].max() for df in property_dfs.values()]
        min_price = min(min_prices)
        max_price = max(max_prices)
    
    # Ensure min_price is not zero to avoid division by zero
    min_price = max(0.1, min_price)  # Using 0.1 as a small positive value
    
    property_prices = np.linspace(min_price, max_price, num_points)
    
    # Color palette for different models
    model_colors = plt.cm.tab10(np.linspace(0, 1, len(impacts)))
    
    # Plot lines for each model
    abs_lines, perc_lines = [], []
    for i, (model_name, impact) in enumerate(impacts.items()):
        # Calculate absolute and percentage values
        impact_value = impact["absolute_impact"]
        absolute_values = [impact_value] * num_points
        
        # Safe division to avoid zero division errors
        percent_values = [impact_value / max(0.1, price) * 100 for price in property_prices]
        
        # Create DataFrame for plotting
        plot_data = pd.DataFrame({
            'Property Price': property_prices,
            'Absolute Impact': absolute_values,
            'Percentage Impact': percent_values
        })
        
        # Plot absolute impact line (in upper subplot)
        abs_line = ax1.plot(property_prices, absolute_values, 
                           color=model_colors[i], linewidth=2.5, marker='o', markersize=6,
                           label=f"{model_name} ({currency}{impact_value:.2f})")
        abs_lines.extend(abs_line)
        
        # Plot percentage impact line (in lower subplot)
        perc_line = ax2.plot(property_prices, percent_values, 
                            color=model_colors[i], linewidth=2.5, marker='x', markersize=6, 
                            label=f"{model_name} ({impact['percentage_impact']:.2f}%)")
        perc_lines.extend(perc_line)
    
    # Set axis labels and titles for each subplot
    ax1.set_ylabel(f'Impacto Absoluto ({currency})', fontsize=12)
    ax1.set_title('Impacto Absoluto por Precio de Propiedad', fontsize=12)
    
    ax2.set_xlabel(f'Precio de la Propiedad ({currency})', fontsize=12)
    ax2.set_ylabel('Impacto Porcentual (%)', fontsize=12)
    ax2.set_title('Impacto Porcentual por Precio de Propiedad', fontsize=12)
    
    # Set custom limits if provided
    if x_lim:
        ax1.set_xlim(x_lim)
        ax2.set_xlim(x_lim)
    if y_lim:
        ax1.set_ylim(y_lim)
    
    # Configure x-axis ticks (only needed for bottom subplot since sharex=True)
    if fixed_x_ticks is not None:
        ax2.set_xticks(fixed_x_ticks)
        tick_labels = [f"{currency}{x/1000:.0f}k" for x in fixed_x_ticks]
        ax2.set_xticklabels(tick_labels, rotation=45)
    else:
        step = max(1, num_points // 5)
        price_labels = [f"{currency}{price/1000:.0f}k" for price in property_prices]
        ax2.set_xticks(property_prices[::step])
        ax2.set_xticklabels(price_labels[::step], rotation=45)
    
    # Configure y-axis ticks
    if fixed_y1_ticks is not None:
        ax1.set_yticks(fixed_y1_ticks)
    if fixed_y2_ticks is not None:
        ax2.set_yticks(fixed_y2_ticks)
    
    # Set overall title for the figure
    if custom_title:
        title = custom_title
    else:
        title = f'Comparación del Impacto de {feature_name} en el Valor de la Propiedad'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    # Create legends for each subplot
    ax1.legend(loc='best', title='Modelos (Valor Absoluto)', frameon=True, framealpha=0.9)
    ax2.legend(loc='best', title='Modelos (Valor Porcentual)', frameon=True, framealpha=0.9)
    
    # Add grid and other styling to both subplots
    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        sns.despine(ax=ax, left=False, bottom=False)
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.9)  # Make space for the overall title
    plt.show()
    
    return {
        "fig": fig,
        "ax1": ax1,
        "ax2": ax2
    }



def plot_regularization_path(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    regularization: str = 'l2',
    alphas: np.ndarray = None,
    method: str = 'gradient_descent',
    learning_rate: float = 0.001,
    epochs: int = 1000,
    figsize: tuple = (18, 6),
    print_metrics: bool = False,
    style: str = 'whitegrid',
    model: Model = LinearRegressor,
    transform_target = None,
    seed: int = 42,
    cv_folds: int = 5,
    show_plots: bool = False,
    plot_types: list = ['coefs', 'cv', 'val', 'combined'],
    metrics: list = ['mse', 'r2']
) -> tuple:
    """
    Visualiza el camino de regularización y métricas de evaluación en función de alpha.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Datos de entrenamiento
    X_test : pd.DataFrame
        Datos de prueba
    y_train : pd.Series
        Variable objetivo de entrenamiento
    y_test : pd.Series
        Variable objetivo de prueba
    regularization : str, default='l2'
        Tipo de regularización ('l1' o 'l2')
    alphas : np.ndarray, default=None
        Lista de valores alpha para evaluar
    method : str, default='gradient_descent'
        Método de entrenamiento del modelo
    learning_rate : float, default=0.001
        Tasa de aprendizaje para descenso de gradiente
    epochs : int, default=1000
        Número de épocas para descenso de gradiente
    figsize : tuple, default=(18, 6)
        Tamaño de figura para los subplots
    print_metrics : bool, default=False
        Si es True, imprime métricas durante el entrenamiento
    style : str, default='whitegrid'
        Estilo de seaborn para las gráficas
    model : Model, default=LinearRegressor
        Clase del modelo a utilizar
    transform_target : callable, default=None
        Función para transformar la variable objetivo
    seed : int, default=42
        Semilla para reproducibilidad
    cv_folds : int, default=5
        Número de folds para validación cruzada
    show_plots : bool, default=False
        Si es True, muestra las figuras inmediatamente
    plot_types : list, default=['coefs', 'cv', 'val', 'combined']
        Lista de gráficos a generar. Opciones:
        - 'coefs': Camino de regularización (coeficientes vs alpha)
        - 'cv': Error por validación cruzada vs alpha
        - 'val': Error en conjunto de validación vs alpha
        - 'combined': Los tres gráficos en una sola figura
        - 'coefs+cv': Coeficientes y error CV en una figura
        - 'coefs+val': Coeficientes y error validación en una figura
        - 'cv+val': Error CV y error validación en una figura
    metrics : list, default=['mse', 'r2']
        Lista de métricas a calcular y visualizar. Opciones:
        - 'mse': Error cuadrático medio
        - 'rmse': Raíz del error cuadrático medio
        - 'r2': Coeficiente de determinación
        - 'mae': Error absoluto medio
    
    Returns
    -------
    tuple
        (figuras, coeficientes, cv_scores, best_metrics)
        figuras: dict con las figuras individuales para cada tipo de gráfico
        coeficientes: array con los coeficientes para cada valor de alpha
        cv_scores: dict con los puntajes de validación cruzada
        best_metrics: dict con métricas y mejores valores de alpha
    """
    # Inicializar y validar parámetros
    alphas = _initialize_alphas(alphas)
    metrics = _validate_metrics(metrics)
    
    # Preparar datos
    feature_names = X_train.columns
    y_train_transformed, y_test_transformed = _prepare_target_data(y_train, y_test, transform_target)
    
    # Crear índices para validación cruzada
    cv_indices = _create_cv_indices(len(X_train), cv_folds, seed)
    
    # Entrenar modelos y calcular métricas
    coefs_array, cv_scores, validation_metrics = _train_models_and_evaluate(
        X_train, X_test, y_train_transformed, y_test_transformed,
        model, method, regularization, alphas, epochs, learning_rate, cv_folds, cv_indices,
        metrics, print_metrics
    )
    
    # Calcular métricas óptimas
    best_metrics = _calculate_best_metrics(alphas, cv_scores, validation_metrics, metrics)
    
    # Crear visualizaciones
    figures = _create_visualizations(
        coefs_array, feature_names, alphas, cv_scores, validation_metrics,
        best_metrics, metrics, regularization, cv_folds, style, figsize, plot_types
    )
    
    # Mostrar plots si se solicita
    if show_plots:
        plt.show()
    else:
        plt.close('all')
    
    return figures, coefs_array, cv_scores, best_metrics


def _initialize_alphas(alphas):
    """Inicializa el array de valores alpha si no se proporciona."""
    if alphas is None:
        return np.linspace(0, 100, 100)
    return alphas


def _validate_metrics(metrics):
    """Valida que las métricas especificadas sean permitidas."""
    allowed_metrics = ['mse', 'rmse', 'r2', 'mae']
    metrics = [m.lower() for m in metrics]
    
    for metric in metrics:
        if metric not in allowed_metrics:
            raise ValueError(f"Métrica no reconocida: {metric}. Métricas disponibles: {allowed_metrics}")
    
    return metrics


def _prepare_target_data(y_train, y_test, transform_target):
    """Aplica transformación a los datos objetivo si se proporciona una función."""
    y_train_transformed = transform_target(y_train) if transform_target else y_train
    y_test_transformed = transform_target(y_test) if transform_target else y_test
    return y_train_transformed, y_test_transformed


def _create_cv_indices(n_samples, cv_folds, seed):
    """Crea índices para validación cruzada."""
    fold_size = n_samples // cv_folds
    indices = np.arange(n_samples)
    np.random.seed(seed=seed)
    np.random.shuffle(indices)
    return indices


def _calculate_metrics(y_true, y_pred, metrics_list):
    """Calcula las métricas solicitadas."""
    results = {}
    if 'mse' in metrics_list:
        results['mse'] = np.mean((y_true - y_pred) ** 2)
    if 'rmse' in metrics_list:
        results['rmse'] = np.sqrt(np.mean((y_true - y_pred) ** 2))
    if 'r2' in metrics_list:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        results['r2'] = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    if 'mae' in metrics_list:
        results['mae'] = np.mean(np.abs(y_true - y_pred))
    return results


def _train_models_and_evaluate(X_train, X_test, y_train, y_test, model_class, 
                              method, regularization, alphas, epochs, learning_rate,
                              cv_folds, indices, metrics, print_metrics):
    """Entrena modelos con diferentes valores de alpha y evalúa su rendimiento."""
    fold_size = len(X_train) // cv_folds
    coefs = []
    cv_scores = {metric: {alpha: [] for alpha in alphas} for metric in metrics}
    validation_metrics = {metric: [] for metric in metrics}
    
    for alpha in alphas:
        # Entrenar modelo sobre todo X_train para coeficientes
        model_full = model_class()
        model_full.fit(
            X_train, y_train, method=method, 
            regularization=regularization, alpha=alpha,
            epochs=epochs, learning_rate=learning_rate
        )
        
        # Guardar coeficientes
        coefs.append(model_full.get_coef_array())
        
        # Evaluar en conjunto de validación
        y_pred_val = model_full.predict(X_test)
        val_metrics = _calculate_metrics(y_test, y_pred_val, metrics)
        
        for metric_name, metric_value in val_metrics.items():
            validation_metrics[metric_name].append(metric_value)
        
        # Ejecutar validación cruzada
        fold_metrics = _cross_validate_model(
            X_train, y_train, model_class, method, regularization,
            alpha, epochs, learning_rate, cv_folds, indices, metrics
        )
        
        # Almacenar scores medios de CV
        for metric in metrics:
            cv_scores[metric][alpha] = np.mean(fold_metrics[metric])
        
        if print_metrics:
            _print_metrics_summary(alpha, cv_scores, validation_metrics, metrics)
    
    return np.array(coefs), cv_scores, validation_metrics


def _cross_validate_model(X_train, y_train, model_class, method, regularization,
                         alpha, epochs, learning_rate, cv_folds, indices, metrics):
    """Ejecuta validación cruzada para un modelo con un valor específico de alpha."""
    fold_size = len(X_train) // cv_folds
    fold_metrics = {metric: [] for metric in metrics}
    
    for fold in range(cv_folds):
        # Crear división train/val para este fold
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([
            indices[:fold * fold_size],
            indices[(fold + 1) * fold_size:]
        ])
        
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]
        
        # Entrenar modelo en este fold
        model_fold = model_class()
        model_fold.fit(
            X_fold_train, y_fold_train, method=method,
            regularization=regularization, alpha=alpha,
            epochs=epochs, learning_rate=learning_rate
        )
        
        # Calcular métricas para este fold
        y_fold_pred = model_fold.predict(X_fold_val)
        fold_results = _calculate_metrics(y_fold_val, y_fold_pred, metrics)
        
        # Almacenar resultados del fold
        for metric_name, metric_value in fold_results.items():
            fold_metrics[metric_name].append(metric_value)
    
    return fold_metrics


def _print_metrics_summary(alpha, cv_scores, validation_metrics, metrics):
    """Imprime un resumen de las métricas para un valor específico de alpha."""
    metric_strings = []
    for metric in metrics:
        metric_strings.append(f"CV {metric.upper()}: {cv_scores[metric][alpha]:.4f}")
        metric_strings.append(f"Val {metric.upper()}: {validation_metrics[metric][-1]:.4f}")
    print(f"Alpha: {alpha:.4f}, " + ", ".join(metric_strings))


def _calculate_best_metrics(alphas, cv_scores, validation_metrics, metrics):
    """Calcula las mejores métricas y sus correspondientes valores de alpha."""
    best_metrics = {}
    
    for metric in metrics:
        cv_values = list(cv_scores[metric].values())
        val_values = validation_metrics[metric]
        
        # Para MSE, RMSE, MAE: el mínimo es mejor
        # Para R²: el máximo es mejor
        if metric in ['mse', 'rmse', 'mae']:
            best_alpha_cv = alphas[np.argmin(cv_values)]
            best_alpha_val = alphas[np.argmin(val_values)]
            best_cv_value = min(cv_values)
            best_val_value = min(val_values)
        else:  # 'r2'
            best_alpha_cv = alphas[np.argmax(cv_values)]
            best_alpha_val = alphas[np.argmax(val_values)]
            best_cv_value = max(cv_values)
            best_val_value = max(val_values)
        
        best_metrics[f'best_alpha_cv_{metric}'] = best_alpha_cv
        best_metrics[f'best_{metric}_cv'] = best_cv_value
        best_metrics[f'best_alpha_val_{metric}'] = best_alpha_val
        best_metrics[f'best_{metric}_val'] = best_val_value
    
    return best_metrics


def _create_visualizations(coefs_array, feature_names, alphas, cv_scores, validation_metrics,
                          best_metrics, metrics, regularization, cv_folds, style, figsize, plot_types):
    """Crea todas las visualizaciones solicitadas."""
    # Preparar datos para gráficos
    plot_data = pd.DataFrame(coefs_array, columns=feature_names)
    plot_data['alpha'] = alphas
    plot_data_melted = pd.melt(plot_data, id_vars=['alpha'], 
                             var_name='Feature', value_name='Coefficient')
    
    # Configurar estilo
    sns.set_style(style)
    
    # Crear diccionario para almacenar figuras
    figures = {}
    
    # Definir funciones de ayuda para crear gráficos
    plot_functions = {
        'create_coefs_plot': _create_coefs_plot,
        'create_metric_plot': _create_metric_plot
    }
    
    # Crear figuras individuales
    figures = _create_individual_plots(
        figures, plot_types, figsize, plot_data_melted, cv_scores, validation_metrics,
        best_metrics, metrics, alphas, regularization, cv_folds, plot_functions
    )
    
    # Crear figuras combinadas
    figures = _create_combined_plots(
        figures, plot_types, figsize, plot_data_melted, cv_scores, validation_metrics,
        best_metrics, metrics, alphas, regularization, cv_folds, plot_functions
    )
    
    # Aplicar tight_layout a todas las figuras
    for fig_dict in figures.values():
        if 'fig' in fig_dict:
            fig_dict['fig'].tight_layout()
    
    return figures


def _create_individual_plots(figures, plot_types, figsize, plot_data_melted, cv_scores, 
                           validation_metrics, best_metrics, metrics, alphas, regularization, 
                           cv_folds, plot_functions):
    """Crea gráficas individuales para coeficientes y métricas."""
    create_coefs_plot = plot_functions['create_coefs_plot']
    create_metric_plot = plot_functions['create_metric_plot']
    
    # Crear gráfico de coeficientes
    if 'coefs' in plot_types:
        fig_coefs = plt.figure(figsize=(figsize[0]//3, figsize[1]))
        ax_coefs = fig_coefs.add_subplot(1, 1, 1)
        create_coefs_plot(ax_coefs, plot_data_melted, regularization)
        figures['coefs'] = {'fig': fig_coefs, 'ax': ax_coefs}
    
    # Crear gráficos individuales para cada métrica
    for metric in metrics:
        # Figura para CV
        if f'cv_{metric}' in plot_types or 'cv' in plot_types:
            fig_cv = plt.figure(figsize=(figsize[0]//3, figsize[1]))
            ax_cv = fig_cv.add_subplot(1, 1, 1)
            create_metric_plot(ax_cv, metric, alphas, cv_scores, validation_metrics, 
                             best_metrics, is_validation=False, cv_folds=cv_folds)
            figures[f'cv_{metric}'] = {'fig': fig_cv, 'ax': ax_cv}
        
        # Figura para validación
        if f'val_{metric}' in plot_types or 'val' in plot_types:
            fig_val = plt.figure(figsize=(figsize[0]//3, figsize[1]))
            ax_val = fig_val.add_subplot(1, 1, 1)
            create_metric_plot(ax_val, metric, alphas, cv_scores, validation_metrics, 
                             best_metrics, is_validation=True, cv_folds=cv_folds)
            figures[f'val_{metric}'] = {'fig': fig_val, 'ax': ax_val}
    
    return figures


def _create_combined_plots(figures, plot_types, figsize, plot_data_melted, cv_scores, 
                          validation_metrics, best_metrics, metrics, alphas, regularization, 
                          cv_folds, plot_functions):
    """Crea gráficos combinados según las opciones solicitadas."""
    create_coefs_plot = plot_functions['create_coefs_plot']
    create_metric_plot = plot_functions['create_metric_plot']
    
    # Crear combinaciones predefinidas
    predefined_combinations = {
        'combined': None,  # Todos los gráficos
        'coefs+cv': ['coefs'] + [f'cv_{m}' for m in metrics],
        'coefs+val': ['coefs'] + [f'val_{m}' for m in metrics],
        'cv+val': sum([[f'cv_{m}', f'val_{m}'] for m in metrics], [])
    }
    
    for plot_type in plot_types:
        if plot_type in predefined_combinations:
            if plot_type == 'combined':
                # Crear figura combinada con todas las métricas
                n_plots = 1 + 2 * len(metrics)  # 1 para coeficientes + 2 por métrica
                fig_combined, axes = plt.subplots(1, n_plots, figsize=(figsize[0]/3*n_plots, figsize[1]))
                
                create_coefs_plot(axes[0], plot_data_melted, regularization)
                
                for i, metric in enumerate(metrics):
                    create_metric_plot(axes[2*i+1], metric, alphas, cv_scores, validation_metrics, 
                                     best_metrics, is_validation=False, cv_folds=cv_folds)
                    create_metric_plot(axes[2*i+2], metric, alphas, cv_scores, validation_metrics, 
                                     best_metrics, is_validation=True, cv_folds=cv_folds)
                
                figures['combined'] = {'fig': fig_combined, 'axes': axes}
            else:
                # Crear otras combinaciones predefinidas
                parts = predefined_combinations[plot_type]
                fig_combined, axes = plt.subplots(1, len(parts), 
                                                figsize=(figsize[0]/3*len(parts), figsize[1]))
                
                # Si solo hay una parte, axes no será iterable
                if len(parts) == 1:
                    axes = [axes]
                
                for i, part in enumerate(parts):
                    if part == 'coefs':
                        create_coefs_plot(axes[i], plot_data_melted, regularization)
                    elif part.startswith('cv_'):
                        metric = part[3:]
                        create_metric_plot(axes[i], metric, alphas, cv_scores, validation_metrics, 
                                         best_metrics, is_validation=False, cv_folds=cv_folds)
                    elif part.startswith('val_'):
                        metric = part[4:]
                        create_metric_plot(axes[i], metric, alphas, cv_scores, validation_metrics, 
                                         best_metrics, is_validation=True, cv_folds=cv_folds)
                
                figures[plot_type] = {'fig': fig_combined, 'axes': axes}
        
        # Procesar combinaciones personalizadas
        elif '+' in plot_type and plot_type not in predefined_combinations:
            _create_custom_combined_plot(
                figures, plot_type, figsize, metrics, plot_data_melted,
                cv_scores, validation_metrics, best_metrics, alphas,
                regularization, cv_folds, plot_functions
            )
    
    return figures


def _create_custom_combined_plot(figures, plot_type, figsize, metrics, plot_data_melted,
                               cv_scores, validation_metrics, best_metrics, alphas,
                               regularization, cv_folds, plot_functions):
    """Crea un gráfico combinado personalizado a partir de partes especificadas."""
    create_coefs_plot = plot_functions['create_coefs_plot']
    create_metric_plot = plot_functions['create_metric_plot']
    
    parts = plot_type.split('+')
    
    # Validar que todas las partes existan
    valid_parts = []
    for part in parts:
        if part == 'coefs':
            valid_parts.append(part)
        elif part.startswith('cv_'):
            metric = part[3:]
            if metric in metrics:
                valid_parts.append(part)
        elif part.startswith('val_'):
            metric = part[4:]
            if metric in metrics:
                valid_parts.append(part)
    
    if valid_parts:
        fig_combined, axes = plt.subplots(1, len(valid_parts), 
                                        figsize=(figsize[0]/3*len(valid_parts), figsize[1]))
        
        # Si solo hay una parte válida, axes no será iterable
        if len(valid_parts) == 1:
            axes = [axes]
        
        for i, part in enumerate(valid_parts):
            if part == 'coefs':
                create_coefs_plot(axes[i], plot_data_melted, regularization)
            elif part.startswith('cv_'):
                metric = part[3:]
                create_metric_plot(axes[i], metric, alphas, cv_scores, validation_metrics, 
                                 best_metrics, is_validation=False, cv_folds=cv_folds)
            elif part.startswith('val_'):
                metric = part[4:]
                create_metric_plot(axes[i], metric, alphas, cv_scores, validation_metrics, 
                                 best_metrics, is_validation=True, cv_folds=cv_folds)
        
        figures[plot_type] = {'fig': fig_combined, 'axes': axes}
    
    return figures


def _create_coefs_plot(ax, plot_data_melted, regularization):
    """Crea un gráfico de coeficientes vs alpha."""
    sns.lineplot(data=plot_data_melted, x='alpha', y='Coefficient', 
                hue='Feature', linewidth=2, ax=ax)
    
    ax.set_xlabel('alpha (regularización)')
    ax.set_ylabel('Valor del peso')
    ax.set_title(f'Coeficientes de {"Lasso" if regularization == "l1" else "Ridge"}\n' 
                 'en función del parámetro de regularización')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Mover la leyenda fuera del gráfico para evitar superposiciones
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


def _create_metric_plot(ax, metric_name, alphas, cv_scores, validation_metrics, 
                       best_metrics, is_validation=False, cv_folds=5):
    """Crea un gráfico de métricas vs alpha."""
    values = validation_metrics[metric_name] if is_validation else list(cv_scores[metric_name].values())
    best_alpha = best_metrics[f'best_alpha_val_{metric_name}'] if is_validation else best_metrics[f'best_alpha_cv_{metric_name}']
    best_value = best_metrics[f'best_{metric_name}_val'] if is_validation else best_metrics[f'best_{metric_name}_cv']
    
    # Para R², el mejor valor es el máximo; para otras métricas, el mínimo
    if metric_name == 'r2':
        best_idx = np.argmax(values)
        value_label = "Mayor"
    else:
        best_idx = np.argmin(values)
        value_label = "Menor"
    
    # Diferentes colores para CV y validación
    color = 'green' if is_validation else 'blue'
    
    ax.plot(alphas, values, '-o', color=color)
    ax.axvline(x=best_alpha, color='r', linestyle='--', 
              label=f'Mejor α={best_alpha:.4f}\n{value_label} {metric_name.upper()}={best_value:.4f}')
    
    ax.set_xlabel('alpha (regularización)')
    
    # Nombres descriptivos para las métricas
    metric_labels = {
        'mse': 'Error Cuadrático Medio',
        'rmse': 'Raíz del Error Cuadrático Medio',
        'r2': 'Coeficiente de Determinación (R²)',
        'mae': 'Error Absoluto Medio'
    }
    
    ax.set_ylabel(metric_labels.get(metric_name, metric_name.upper()))
    
    # Indicar si es validación cruzada o conjunto de validación
    source = "Conjunto de Validación" if is_validation else f"Validación Cruzada ({cv_folds}-fold)"
    ax.set_title(f'{metric_labels.get(metric_name, metric_name.upper())}\n{source}')
    
    ax.grid(True, alpha=0.3)
    ax.legend()