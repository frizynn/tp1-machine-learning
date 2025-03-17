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



def plot_regularization_path( # hacer que devuelva el mejor modelo 
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
    metrics: list = ['mse']
) -> tuple:
    """
    Visualiza el camino de regularización y las métricas seleccionadas en varios subplots:
    1) Coeficientes vs. alpha
    2) Métrica(s) por validación cruzada vs. alpha (en subplots separados)
    3) Métrica(s) en conjunto de validación vs. alpha (en subplots separados)

    Parámetros
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
        - 'cv': Métrica por validación cruzada vs alpha
        - 'val': Métrica en conjunto de validación vs alpha
        - 'combined': Los tres gráficos en una sola figura
        - 'coefs+cv': Coeficientes y error CV en una figura
        - 'coefs+val': Coeficientes y error validación en una figura
        - 'cv+val': Error CV y error validación en una figura
    metrics : list, default=['mse']
        Lista de métricas a utilizar. Opciones: 'mse', 'r2' o ambas.

    Returns
    -------
    tuple
        (figuras, coeficientes, cv_scores, best_metrics)
        figuras: dict con las figuras individuales para cada tipo de gráfico
        coeficientes: array con los coeficientes para cada valor de alpha
        cv_scores: dict con los puntajes de validación cruzada para cada métrica
        best_metrics: dict con las métricas y mejores valores de alpha para cada métrica
    """
    if alphas is None:
        alphas = np.linspace(0, 100, 100)
    
    # Validar métricas
    valid_metrics = ['mse', 'r2']
    for m in metrics:
        if m not in valid_metrics:
            raise ValueError(f"Unsupported metric '{m}': choose from {valid_metrics}")
    
    # Definir etiquetas y criterios según las métricas
    metric_labels = {
        'mse': 'Error Cuadrático Medio (promedio de folds)',
        'r2': 'Coeficiente de Determinación (R^2)'
    }
    better_is_lower = {
        'mse': True,
        'r2': False
    }
    
    feature_names = X_train.columns
    coefs = []
    cv_scores = {metric: {alpha: [] for alpha in alphas} for metric in metrics}
    validation_metrics = {metric: [] for metric in metrics}
    
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
        
        # Calcular todas las métricas solicitadas
        for metric in metrics:
            if metric == 'mse':
                val_score = model_full.mse_score(X_test, y_test_transformed)
            elif metric == 'r2':
                val_score = 1 - ((y_test_transformed - y_pred_val) ** 2).sum() / ((y_test_transformed - y_test_transformed.mean()) ** 2).sum()
            
            validation_metrics[metric].append(val_score)
        
        # 2. Ejecutar validación cruzada para obtener métricas robustas
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
            
            # Calculate metrics for this fold
            y_pred_fold = model_fold.predict(X_fold_val)
            
            for metric in metrics:
                if metric == 'mse':
                    score = model_fold.mse_score(X_fold_val, y_fold_val)
                elif metric == 'r2':
                    score = 1 - ((y_fold_val - y_pred_fold) ** 2).sum() / ((y_fold_val - y_fold_val.mean()) ** 2).sum()
                
                if fold == 0:
                    cv_scores[metric][alpha] = [score]
                else:
                    cv_scores[metric][alpha].append(score)
        
        if print_metrics:
            metric_strings = []
            for metric in metrics:
                cv_mean = np.mean(cv_scores[metric][alpha])
                val_score = validation_metrics[metric][-1]
                metric_strings.append(f"CV {metric_labels[metric]}: {cv_mean:.4f}, Val {metric_labels[metric]}: {val_score:.4f}")
            print(f"Alpha: {alpha:.4f}, " + ", ".join(metric_strings))
    
    # Convert coefficients to array
    coefs_array = np.array(coefs)
    
    # Calcular valores medios de validación cruzada
    cv_values = {metric: [np.mean(cv_scores[metric][alpha]) for alpha in alphas] for metric in metrics}
    
    # Determinar mejores alphas para cada métrica
    best_metrics = {}
    for metric in metrics:
        if better_is_lower[metric]:
            best_alpha_cv = alphas[np.argmin(cv_values[metric])]
            best_cv_value = min(cv_values[metric])
            best_alpha_val = alphas[np.argmin(validation_metrics[metric])]
            best_val_value = min(validation_metrics[metric])
        else:
            best_alpha_cv = alphas[np.argmax(cv_values[metric])]
            best_cv_value = max(cv_values[metric])
            best_alpha_val = alphas[np.argmax(validation_metrics[metric])]
            best_val_value = max(validation_metrics[metric])
        
        best_metrics[metric] = {
            'best_alpha_cv': best_alpha_cv,
            'best_cv_score': best_cv_value,
            'best_alpha_val': best_alpha_val,
            'best_val_score': best_val_value
        }
    
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
    
    def create_cv_plot(fig, metric_index, num_metrics):
        """Crear un subplot para métricas de CV"""
        metric = metrics[metric_index]
        ax = fig.add_subplot(1, num_metrics, metric_index + 1)
        
        color = f'C{metric_index}'
        ax.plot(alphas, cv_values[metric], '-o', color=color, linewidth=2, 
                label=f'{metric_labels[metric]} (CV)')
        
        best_alpha = best_metrics[metric]['best_alpha_cv']
        best_score = best_metrics[metric]['best_cv_score']
        ax.axvline(x=best_alpha, color=color, linestyle='--', 
                label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
        
        ax.set_xlabel('alpha (regularización)')
        ax.set_ylabel(f'{metric_labels[metric]}')
        ax.set_title(f'{metric_labels[metric]} por CV ({cv_folds}-fold)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    def create_val_plot(fig, metric_index, num_metrics):
        """Crear un subplot para métricas de validación"""
        metric = metrics[metric_index]
        ax = fig.add_subplot(1, num_metrics, metric_index + 1)
        
        color = f'C{metric_index + 2}'  # Usar colores diferentes a los de CV
        ax.plot(alphas, validation_metrics[metric], '-o', color=color, linewidth=2,
            label=f'{metric_labels[metric]} (Val)')
        
        best_alpha = best_metrics[metric]['best_alpha_val']
        best_score = best_metrics[metric]['best_val_score']
        ax.axvline(x=best_alpha, color=color, linestyle='--', 
                label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
        
        ax.set_xlabel('alpha (regularización)')
        ax.set_ylabel(f'{metric_labels[metric]}')
        ax.set_title(f'{metric_labels[metric]} en Validación')
        ax.grid(True, alpha=0.3)
        ax.legend()
        return ax

    # Crear figuras individuales
    if 'coefs' in plot_types:
        fig_coefs = plt.figure(figsize=(figsize[0]//3, figsize[1]))
        ax_coefs = fig_coefs.add_subplot(1, 1, 1)
        create_coefs_plot(ax_coefs)
        figures['coefs'] = {'fig': fig_coefs, 'ax': ax_coefs}

    if 'cv' in plot_types:
        # Cambio: usar layout horizontal (1 fila, múltiples columnas)
        fig_cv = plt.figure(figsize=(figsize[0], figsize[1]//2))
        axes_cv = []
        for i in range(len(metrics)):
            ax = create_cv_plot(fig_cv, i, len(metrics))
            axes_cv.append(ax)
        figures['cv'] = {'fig': fig_cv, 'axes': axes_cv}

    if 'val' in plot_types:
        # Cambio: usar layout horizontal (1 fila, múltiples columnas)
        fig_val = plt.figure(figsize=(figsize[0], figsize[1]//2))
        axes_val = []
        for i in range(len(metrics)):
            ax = create_val_plot(fig_val, i, len(metrics))
            axes_val.append(ax)
        figures['val'] = {'fig': fig_val, 'axes': axes_val}

    
    if 'coefs+cv' in plot_types:
        # Cambio: layout horizontal para métricas CV
        fig_coefs_cv = plt.figure(figsize=figsize)
        gs = fig_coefs_cv.add_gridspec(2, 1)
        
        # Plot coefs (ocupa toda la primera fila)
        ax_coefs = fig_coefs_cv.add_subplot(gs[0, 0])
        create_coefs_plot(ax_coefs)
        
        # Plot CV metrics (segunda fila, una columna por métrica)
        if len(metrics) == 1:
            axes_cv = []
            metric = metrics[0]
            ax = fig_coefs_cv.add_subplot(gs[1, 0])
            color = 'C0'
            ax.plot(alphas, cv_values[metric], '-o', color=color, linewidth=2, 
                    label=f'{metric_labels[metric]} (CV)')
            
            best_alpha = best_metrics[metric]['best_alpha_cv']
            best_score = best_metrics[metric]['best_cv_score']
            ax.axvline(x=best_alpha, color=color, linestyle='--', 
                    label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
            
            ax.set_xlabel('alpha (regularización)')
            ax.set_ylabel(f'{metric_labels[metric]}')
            ax.set_title(f'{metric_labels[metric]} por CV ({cv_folds}-fold)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            axes_cv.append(ax)
        else:
            # Crear subplots horizontales para múltiples métricas
            axes_cv = []
            gs_metrics = gs[1, 0].subgridspec(1, len(metrics))
            for i in range(len(metrics)):
                ax = fig_coefs_cv.add_subplot(gs_metrics[0, i])
                metric = metrics[i]
                color = f'C{i}'
                ax.plot(alphas, cv_values[metric], '-o', color=color, linewidth=2, 
                        label=f'{metric_labels[metric]} (CV)')
                
                best_alpha = best_metrics[metric]['best_alpha_cv']
                best_score = best_metrics[metric]['best_cv_score']
                ax.axvline(x=best_alpha, color=color, linestyle='--', 
                        label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
                
                ax.set_xlabel('alpha (regularización)')
                ax.set_ylabel(f'{metric_labels[metric]}')
                ax.set_title(f'{metric_labels[metric]} por CV ({cv_folds}-fold)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                axes_cv.append(ax)
        
        figures['coefs+cv'] = {'fig': fig_coefs_cv, 'ax_coefs': ax_coefs, 'axes_cv': axes_cv}

    if 'coefs+val' in plot_types:
        # Cambio: layout horizontal para métricas de validación
        fig_coefs_val = plt.figure(figsize=figsize)
        gs = fig_coefs_val.add_gridspec(2, 1)
        
        # Plot coefs (ocupa toda la primera fila)
        ax_coefs = fig_coefs_val.add_subplot(gs[0, 0])
        create_coefs_plot(ax_coefs)
        
        # Plot validation metrics (segunda fila, una columna por métrica)
        if len(metrics) == 1:
            axes_val = []
            metric = metrics[0]
            ax = fig_coefs_val.add_subplot(gs[1, 0])
            color = 'C2'  # Color distinto al de CV
            ax.plot(alphas, validation_metrics[metric], '-o', color=color, linewidth=2,
                label=f'{metric_labels[metric]} (Val)')
            
            best_alpha = best_metrics[metric]['best_alpha_val']
            best_score = best_metrics[metric]['best_val_score']
            ax.axvline(x=best_alpha, color=color, linestyle='--', 
                    label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
            
            ax.set_xlabel('alpha (regularización)')
            ax.set_ylabel(f'{metric_labels[metric]}')
            ax.set_title(f'{metric_labels[metric]} en Validación')
            ax.grid(True, alpha=0.3)
            ax.legend()
            axes_val.append(ax)
        else:
            # Crear subplots horizontales para múltiples métricas
            axes_val = []
            gs_metrics = gs[1, 0].subgridspec(1, len(metrics))
            for i in range(len(metrics)):
                ax = fig_coefs_val.add_subplot(gs_metrics[0, i])
                metric = metrics[i]
                color = f'C{i+2}'  # Color distinto al de CV
                ax.plot(alphas, validation_metrics[metric], '-o', color=color, linewidth=2,
                    label=f'{metric_labels[metric]} (Val)')
                
                best_alpha = best_metrics[metric]['best_alpha_val']
                best_score = best_metrics[metric]['best_val_score']
                ax.axvline(x=best_alpha, color=color, linestyle='--', 
                        label=f'Mejor α={best_alpha:.4f}\n{metric_labels[metric]}={best_score:.4f}')
                
                ax.set_xlabel('alpha (regularización)')
                ax.set_ylabel(f'{metric_labels[metric]}')
                ax.set_title(f'{metric_labels[metric]} en Validación')
                ax.grid(True, alpha=0.3)
                ax.legend()
                axes_val.append(ax)
        
        figures['coefs+val'] = {'fig': fig_coefs_val, 'ax_coefs': ax_coefs, 'axes_val': axes_val}


        
    # Aplicar tight_layout a todas las figuras
    for fig_dict in figures.values():
        if 'fig' in fig_dict:
            fig_dict['fig'].tight_layout()
    
    # Mostrar plots si se solicitó
    if show_plots:
        plt.show()
    
    return figures, coefs_array, cv_scores, best_metrics