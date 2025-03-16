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
    plot_types: list = ['coefs', 'cv', 'val', 'combined']
) -> tuple:
    """
    Visualiza el camino de regularización y el ECM en tres subplots:
    1) Coeficientes vs. alpha
    2) ECM por validación cruzada vs. alpha
    3) ECM en conjunto de validación vs. alpha
    
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
    
    Returns
    -------
    tuple
        (figuras, coeficientes, cv_scores, best_metrics)
        figuras: dict con las figuras individuales para cada tipo de gráfico
        coeficientes: array con los coeficientes para cada valor de alpha
        cv_scores: dict con los puntajes de validación cruzada
        best_metrics: dict con métricas y mejores valores de alpha
    """
    if alphas is None:
        alphas = np.linspace(0, 100, 100)
    
    feature_names = X_train.columns
    coefs = []
    cv_scores = {alpha: [] for alpha in alphas}
    validation_metrics = {
        'mse': [],
        'r2': []
    }
    
    # Transform target if needed
    y_train_transformed = transform_target(y_train) if transform_target else y_train
    y_test_transformed = transform_target(y_test) if transform_target else y_test
    
    # Prepare cross-validation splits
    n_samples = len(X_train)
    fold_size = n_samples // cv_folds
    indices = np.arange(n_samples)
    np.random.seed(seed=seed)
    np.random.shuffle(indices)
    
    # Loop through alphas
    for alpha in alphas:
        # 1. Entrenar modelo sobre X_train completo para obtener coeficientes
        model_full = model()
        model_full.fit(
            X_train,
            y_train_transformed,
            method=method,
            regularization=regularization,
            alpha=alpha,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        # Guardar coeficientes del modelo entrenado sobre X_train completo
        coefs.append(model_full.get_coef_array())
        
        # Evaluar en conjunto de validación (X_test)
        y_pred_val = model_full.predict(X_test)
        mse_val = model_full.mse_score(X_test, y_test_transformed)
        r2_val = 1 - ((y_test_transformed - y_pred_val) ** 2).sum() / ((y_test_transformed - y_test_transformed.mean()) ** 2).sum()
        
        validation_metrics['mse'].append(mse_val)
        validation_metrics['r2'].append(r2_val)
        
        # 2. Ejecutar validación cruzada para obtener ECM robusto
        fold_scores = []
        
        for fold in range(cv_folds):
            # Create train/val split for this fold
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([
                indices[:fold * fold_size],
                indices[(fold + 1) * fold_size:]
            ])
            
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train_transformed.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train_transformed.iloc[val_idx]
            
            # Train model on this fold
            model_fold = model()
            model_fold.fit(
                X_fold_train,
                y_fold_train,
                method=method,
                regularization=regularization,
                alpha=alpha,
                epochs=epochs,
                learning_rate=learning_rate
            )
            
            # Calculate MSE for this fold
            mse = model_fold.mse_score(X_fold_val, y_fold_val)
            fold_scores.append(mse)
        
        # Store mean CV score for this alpha
        cv_scores[alpha] = np.mean(fold_scores)
        
        if print_metrics:
            print(f"Alpha: {alpha:.4f}, CV MSE: {cv_scores[alpha]:.4f}, Validation MSE: {mse_val:.4f}")
    
    # Convert coefficients to array
    coefs_array = np.array(coefs)
    
    # Preparar métricas
    cv_values = list(cv_scores.values())
    validation_mse_array = np.array(validation_metrics['mse'])
    validation_r2_array = np.array(validation_metrics['r2'])
    
    best_alpha_cv = alphas[np.argmin(cv_values)]
    best_alpha_val = alphas[np.argmin(validation_mse_array)]
    
    # Preparar datos para gráficos
    plot_data = pd.DataFrame(coefs_array, columns=feature_names)
    plot_data['alpha'] = alphas
    plot_data_melted = pd.melt(plot_data, id_vars=['alpha'], 
                              var_name='Feature', value_name='Coefficient')
    
    # Configurar estilo
    sns.set_style(style)
    
    # Diccionario para almacenar las figuras
    figures = {}
    
    # Funciones auxiliares para crear gráficos
    def create_coefs_plot(ax):
        sns.lineplot(data=plot_data_melted, x='alpha', y='Coefficient', 
                    hue='Feature', linewidth=2, ax=ax)
        
        ax.set_xlabel('alpha (regularización)')
        ax.set_ylabel('Valor del peso')
        ax.set_title(f'Coeficientes de {"Lasso" if regularization == "l1" else "Ridge"}\n' 
                     'en función del parámetro de regularización')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def create_cv_plot(ax):
        ax.plot(alphas, cv_values, '-o', color='blue')
        ax.axvline(x=best_alpha_cv, color='r', linestyle='--', 
                  label=f'Mejor α={best_alpha_cv:.4f}\nECM={min(cv_values):.4f}')
        
        ax.set_xlabel('alpha (regularización)')
        ax.set_ylabel('Error Cuadrático Medio')
        ax.set_title(f'ECM por Validación Cruzada\n{cv_folds}-fold CV')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def create_val_plot(ax):
        ax.plot(alphas, validation_mse_array, '-o', color='green')
        ax.axvline(x=best_alpha_val, color='r', linestyle='--', 
                  label=f'Mejor α={best_alpha_val:.4f}\nECM={min(validation_mse_array):.4f}')
        
        ax.set_xlabel('alpha (regularización)')
        ax.set_ylabel('Error Cuadrático Medio')
        ax.set_title(f'ECM en Conjunto de Validación\nTest Set')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Crear figuras individuales
    if 'coefs' in plot_types:
        fig_coefs = plt.figure(figsize=(figsize[0]//3, figsize[1]))
        ax_coefs = fig_coefs.add_subplot(1, 1, 1)
        create_coefs_plot(ax_coefs)
        figures['coefs'] = {'fig': fig_coefs, 'ax': ax_coefs}
    
    if 'cv' in plot_types:
        fig_cv = plt.figure(figsize=(figsize[0]//3, figsize[1]))
        ax_cv = fig_cv.add_subplot(1, 1, 1)
        create_cv_plot(ax_cv)
        figures['cv'] = {'fig': fig_cv, 'ax': ax_cv}
    
    if 'val' in plot_types:
        fig_val = plt.figure(figsize=(figsize[0]//3, figsize[1]))
        ax_val = fig_val.add_subplot(1, 1, 1)
        create_val_plot(ax_val)
        figures['val'] = {'fig': fig_val, 'ax': ax_val}
    
    # Crear figuras combinadas
    if 'combined' in plot_types:
        fig_combined, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        create_coefs_plot(ax1)
        create_cv_plot(ax2)
        create_val_plot(ax3)
        figures['combined'] = {'fig': fig_combined, 'axes': [ax1, ax2, ax3]}
    
    # Combinaciones adicionales
    if 'coefs+cv' in plot_types:
        fig_coefs_cv, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]//3*2, figsize[1]))
        create_coefs_plot(ax1)
        create_cv_plot(ax2)
        figures['coefs+cv'] = {'fig': fig_coefs_cv, 'axes': [ax1, ax2]}
    
    if 'coefs+val' in plot_types:
        fig_coefs_val, (ax1, ax2) = plt.subplots(1, 2, figsize=(figsize[0]//3*2, figsize[1]))
        create_coefs_plot(ax1)
        create_val_plot(ax2)
        figures['coefs+val'] = {'fig': fig_coefs_val, 'axes': [ax1, ax2]}
    

    
    # Aplicar tight_layout a todas las figuras
    for fig_dict in figures.values():
        if 'fig' in fig_dict:
            fig_dict['fig'].tight_layout()
    
    # Mostrar plots si se solicitó
    if show_plots:
        plt.show()
    
    # Find best alpha values
    best_metrics = {
        'best_alpha_cv': best_alpha_cv,
        'best_mse_cv': min(cv_values),
        'best_alpha_val': best_alpha_val,
        'best_mse_val': min(validation_mse_array),
        'best_alpha_r2': alphas[np.argmax(validation_r2_array)],
        'best_r2': max(validation_r2_array)
    }
    
    return figures, coefs_array, cv_scores, best_metrics