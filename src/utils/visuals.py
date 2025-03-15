from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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