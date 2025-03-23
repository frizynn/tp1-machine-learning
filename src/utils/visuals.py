import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def _save_and_show_figure(fig, save_path, filename, show_figures, dpi=300):
    """
    Función auxiliar para manejar el guardado y visualización de figuras
    
    Parámetros:
    -----------
    fig : matplotlib.figure.Figure
        Figura para guardar/mostrar
    save_path : str o None
        Ruta del directorio para guardar la figura
    filename : str
        Nombre de archivo a utilizar al guardar
    show_figures : bool
        Indica si se debe mostrar la figura
    dpi : int
        Resolución para la figura guardada
    
    Retorna:
    --------
    matplotlib.figure.Figure
        La figura de entrada
    """
    if save_path:
        plt.savefig(f"{save_path}/{filename}", dpi=dpi, bbox_inches='tight')
    
    if show_figures:
        plt.show()
    else:
        plt.close()
    
    return fig


def visualize_regression_results(y_true, y_pred, transform_func=None, fig_size=(10, 5), titles=None,
                                 save_path=None, show_figures=True, fit_degree=1):
    """
    Crea visualizaciones completas para evaluar un modelo de regresión.
    
    Parámetros:
    -----------
    y_true : array-like
        Valores reales de la variable objetivo.
    y_pred : array-like
        Valores predichos por el modelo.
    transform_func : callable, opcional
        Función para transformar los valores (por ejemplo, np.exp).
    fig_size : tuple, opcional
        Tamaño de la figura (por defecto (10, 5)).
    titles : dict, opcional
        Títulos personalizados para los gráficos. Claves: 'scatter', 'residuals', 'distribution', 'qq_plot'.
    save_path : str, opcional
        Ruta donde se guardarán las figuras; si es None, no se guardan.
    show_figures : bool, opcional
        Indica si se muestran las figuras.
    fit_degree : int, opcional
        Grado del ajuste polinómico (1=lineal, 2=cuadrático, etc.).
    
    Retorna:
    --------
    dict
        Diccionario con las figuras creadas.
    """
    def _initialize_plot_titles(titles):
        """
        Inicializa y valida los títulos de los gráficos.
        
        Parámetros:
        -----------
        titles : dict
            Títulos personalizados.
        
        Retorna:
        --------
        dict
            Diccionario con todos los títulos necesarios (completando los que falten).
        """
        default_titles = {
            "scatter": "Precio Real vs Precio Predicho",
            "residuals": "Residuos vs Precio Predicho",
            "distribution": "Distribución de Residuos",
            "qq_plot": "Normal Q-Q Plot de Residuos"
        }
        
        if not titles:
            return default_titles
        else:
            # se agregan los títulos faltantes usando los valores por defecto
            for key in default_titles:
                if key not in titles:
                    titles[key] = default_titles[key]
            return titles

    titles = _initialize_plot_titles(titles)
    labels = {"actual": "Precio Real", "predicted": "Precio Predicho", "residuals": "Residuos"}
    
    if transform_func:
        y_true = transform_func(y_true)
        y_pred = transform_func(y_pred)
    
    residuals = y_true - y_pred  # calcular los residuos
    
    figures = {}
    figures["scatter"] = _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, save_path, show_figures, fit_degree)
    figures["residuals"] = _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, save_path, show_figures)
    figures["distribution"] = _create_distribution_plot(residuals, labels, titles, fig_size, save_path, show_figures)
    figures["qq_plot"] = _create_qq_plot(residuals, titles, fig_size, save_path, show_figures)
    
    return figures


def _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, save_path, show_figures, fit_degree):
    """
    Crea un gráfico de dispersión entre los valores reales y predichos, e incluye la línea de predicción perfecta.
    
    Parámetros:
    -----------
    y_true : array-like
        Valores reales.
    y_pred : array-like
        Valores predichos.
    labels : dict
        Etiquetas para los ejes.
    titles : dict
        Títulos de los gráficos.
    fig_size : tuple
        Tamaño de la figura.
    save_path : str
        Ruta para guardar la figura.
    show_figures : bool
        Indica si se muestra la figura.
    fit_degree : int
        Grado del ajuste polinómico; si es mayor que 1 se realiza el ajuste.
    
    Retorna:
    --------
    matplotlib.figure.Figure
        La figura generada.
    """
    fig_scatter = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, label="Datos")
    
    x_data = np.array(y_true)
    y_data = np.array(y_pred)
    
    # determinar el mínimo y máximo para trazar la línea de predicción perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Predicción perfecta")
    
    # si se especifica un grado mayor a 1, se ajusta un polinomio a los datos
    if fit_degree > 1:
        coeffs = np.polyfit(x_data, y_data, fit_degree)  # obtiene los coeficientes del polinomio
        p = np.poly1d(coeffs)  # crea una función polinómica
        
        x_curve = np.linspace(min_val, max_val, 100)  # genera puntos para la curva
        y_curve = p(x_curve)  # evalúa el polinomio en los puntos generados
        
        plt.plot(x_curve, y_curve, 'r-', label=f"Ajuste polinómico (grado={fit_degree})")
    
    plt.xlabel(labels["actual"])
    plt.ylabel(labels["predicted"])
    plt.title(titles["scatter"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    return _save_and_show_figure(fig_scatter, save_path, "scatter_plot.png", show_figures)


def _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, save_path, show_figures):
    """
    Crea un gráfico de dispersión de los residuos frente a los valores predichos.
    
    Parámetros:
    -----------
    y_pred : array-like
        Valores predichos.
    residuals : array-like
        Diferencia entre los valores reales y predichos.
    labels : dict
        Etiquetas para los ejes.
    titles : dict
        Títulos de los gráficos.
    fig_size : tuple
        Tamaño de la figura.
    save_path : str
        Ruta para guardar la figura.
    show_figures : bool
        Indica si se muestra la figura.
    
    Retorna:
    --------
    matplotlib.figure.Figure
        La figura generada.
    """
    fig_residuals = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, label="Residuos")
    
    plt.axhline(y=0, color='r', linestyle='--', label="Residuo cero")  # línea horizontal en cero
    
    plt.xlabel(labels["predicted"])
    plt.ylabel(labels["residuals"])
    plt.title(titles["residuals"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    return _save_and_show_figure(fig_residuals, save_path, "residuals_plot.png", show_figures)


def _create_distribution_plot(residuals, labels, titles, fig_size, save_path, show_figures):
    """
    Crea un histograma de la distribución de los residuos e incluye la curva de una distribución normal teórica.
    
    Parámetros:
    -----------
    residuals : array-like
        Residuos de la regresión.
    labels : dict
        Etiquetas para los ejes.
    titles : dict
        Títulos de los gráficos.
    fig_size : tuple
        Tamaño de la figura.
    save_path : str
        Ruta para guardar la figura.
    show_figures : bool
        Indica si se muestra la figura.
    
    Retorna:
    --------
    matplotlib.figure.Figure
        La figura generada.
    """
    fig_dist = plt.figure(figsize=fig_size)
    
    sns.histplot(residuals, kde=True, stat="density", label="Distribución")
    
    plt.xlabel(labels["residuals"])
    plt.ylabel("Frecuencia")
    plt.title(titles["distribution"])
    plt.grid(True, alpha=0.3, linestyle='--')
    
    x = np.linspace(np.min(residuals), np.max(residuals), 100)
    mean = np.mean(residuals)
    std = np.std(residuals)
    # calcular la función de densidad de probabilidad de una normal
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    plt.plot(x, normal_pdf, 'r-', label="Distribución normal")
    plt.legend()
    
    return _save_and_show_figure(fig_dist, save_path, "residuals_distribution.png", show_figures)


def _create_qq_plot(residuals, titles, fig_size, save_path, show_figures):
    """
    Crea un gráfico Q-Q para comparar los cuantiles de los residuos con los cuantiles teóricos de una distribución normal.
    
    Parámetros:
    -----------
    residuals : array-like
        Residuos de la regresión.
    titles : dict
        Títulos de los gráficos.
    fig_size : tuple
        Tamaño de la figura.
    save_path : str
        Ruta para guardar la figura.
    show_figures : bool
        Indica si se muestra la figura.
    
    Retorna:
    --------
    matplotlib.figure.Figure
        La figura generada.
    """
    def erfinv(y):
        """
        Aproxima la inversa de la función error utilizando la fórmula de Winitzki.
        
        Parámetros:
        -----------
        y : float o array-like
            Valor(es) para los que se calcula la inversa.
        
        Retorna:
        --------
        float o array-like
            Valor(es) aproximado(s) de la inversa de la función error.
        """
        a = 0.147
        ln = np.log(1 - y**2)  # calcular el logaritmo de (1 - y^2)
        term1 = 2/(np.pi * a) + ln/2
        return np.sign(y) * np.sqrt(np.sqrt(term1**2 - ln/a) - term1)  # cálculo de la inversa
        
    fig_qq = plt.figure(figsize=fig_size)
    
    ordered_values = np.sort(residuals)  # ordenar los residuos
    n = len(residuals)
    
    # calcular los porcentajes para cada punto (p = (i-0.5)/n)
    probs = (np.arange(1, n+1) - 0.5) / n
    
    # calcular los cuantiles teóricos usando la función erfinv
    theoretical_quantiles = np.sqrt(2) * erfinv(2 * probs - 1)
    
    # ajustar una línea de regresión lineal entre los cuantiles teóricos y los observados
    slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, 1)
    
    # calcular la correlación entre los cuantiles teóricos y observados
    r = np.corrcoef(theoretical_quantiles, ordered_values)[0, 1]
    
    plt.scatter(theoretical_quantiles, ordered_values, alpha=0.7, label="Cuantiles")
    plt.plot(theoretical_quantiles, slope * theoretical_quantiles + intercept, 'r--',
             label=f"Línea de referencia (r={r:.3f})")
    
    plt.xlabel("Cuantiles teóricos")
    plt.ylabel("Cuantiles observados")
    plt.title(titles["qq_plot"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    return _save_and_show_figure(fig_qq, save_path, "qq_plot.png", show_figures)


def _setup_plot_with_optimal_value(lambdas, metric_scores, metric_name, is_minimize=None, optimal_lambda=None, optimal_metric=None, ax=None):
    """
    función auxiliar para configurar un gráfico con un marcador de valor óptimo
    
    parámetros:
    -----------
    lambdas : array-like
        valores de lambda
    metric_scores : array-like
        puntuaciones para cada lambda
    metric_name : str
        nombre de la métrica
    is_minimize : bool, opcional
        si la métrica debe ser minimizada (si es None, se infiere del nombre de la métrica)
    optimal_lambda : float, opcional
        valor óptimo de lambda pre-calculado
    optimal_metric : float, opcional
        valor óptimo de la métrica pre-calculado
    ax : matplotlib.axes.Axes, opcional
        ejes sobre los que graficar
    
    retorna:
    --------
    tuple
        (optimal_lambda, optimal_metric, ax)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # grafica la línea
    sns.lineplot(x=lambdas, y=metric_scores, ax=ax, color='royalblue', label=metric_name)
    
    # determina si estamos minimizando o maximizando
    if is_minimize is None:
        is_minimize = any(term in metric_name.lower() for term in ['error', 'loss', 'mse', 'mae'])
    
    # encuentra el valor óptimo si no se proporciona
    if optimal_lambda is None or optimal_metric is None:
        if is_minimize:
            optimal_idx = np.argmin(metric_scores)
            optimal_type = "Min"
        else:
            optimal_idx = np.argmax(metric_scores)
            optimal_type = "Max"
        
        optimal_metric = metric_scores[optimal_idx]
        optimal_lambda = lambdas[optimal_idx]
    else:
        optimal_type = "Min" if is_minimize else "Max"
    
    # marca el punto óptimo
    ax.scatter([optimal_lambda], [optimal_metric], color='red', s=100, zorder=5,
               label=f'{optimal_type} {metric_name}: {optimal_metric:.4f} (λ = {optimal_lambda:.4f})')
    ax.axvline(x=optimal_lambda, color='red', linestyle='--', alpha=0.7)
    
    ax.text(0.02, 0.98, f'λ óptimo: {optimal_lambda:.4f}\n{metric_name} {optimal_type.lower()}: {optimal_metric:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    return optimal_lambda, optimal_metric, ax


def plot_weights_vs_lambda(lambdas, weights, feature_names, custom_titles=None):
    """
    Grafica los valores de los pesos en función de la regularización λ usando seaborn.
    
    Parámetros:
    -----------
    lambdas : lista o array
        Valores de λ.
    weights : array-like
        Matriz donde cada columna representa los pesos de una característica para cada λ.
    feature_names : lista
        Nombres de las características.
    custom_titles : dict, opcional
        Títulos personalizados con claves: 'title', 'xlabel', 'ylabel'.
    
    Retorna:
    --------
    None
    """
    # títulos por defecto
    titles = {
        "title": "Ridge Regression: Weight Values vs Regularization Strength",
        "xlabel": "Regularization strength (λ)",
        "ylabel": "Weight Value"
    }
    
    # actualizar con títulos personalizados si se proporcionan
    if custom_titles:
        titles.update(custom_titles)

    plt.figure(figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # crear un DataFrame para facilitar el uso con seaborn
    data = pd.DataFrame(weights, columns=feature_names[:weights.shape[1]])
    data['lambda'] = lambdas
    
    # convertir el DataFrame a formato long para graficar múltiples líneas
    data_long = pd.melt(data, id_vars=['lambda'], var_name='feature', value_name='weight')
    
    ax = sns.lineplot(x='lambda', y='weight', hue='feature', data=data_long)
    
    plt.xlabel(titles["xlabel"], fontsize=12)
    plt.ylabel(titles["ylabel"], fontsize=12)
    plt.title(titles["title"], fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # mover la leyenda fuera del gráfico
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
    
    plt.tight_layout()
    plt.show()


def plot_metric_vs_lambda(ax, lambdas, metric_scores, metric_name="MSE", marker='o'):
    """
    Grafica una métrica (por ejemplo, MSE o R²) en función de λ.
    
    Parámetros:
    -----------
    ax : matplotlib.axes.Axes
        Eje donde se dibuja el gráfico.
    lambdas : lista o array
        Valores de λ.
    metric_scores : lista o array
        Valores de la métrica.
    metric_name : str, opcional
        Nombre de la métrica ("MSE" o "R2"). Por defecto "MSE".
    marker : str, opcional
        Marcador para el punto óptimo.
    
    Retorna:
    --------
    tuple
        (optimal_lambda, optimal_metric) con el λ y la métrica óptimos.
    """
    is_minimize = metric_name.upper() != "R2"
    optimal_lambda, optimal_metric, _ = _setup_plot_with_optimal_value(
        lambdas, metric_scores, metric_name, is_minimize, ax=ax)
    
    ax.set_xlabel('Intensidad de regularización (λ)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs λ')
    ax.grid(True)
    
    return optimal_lambda, optimal_metric


def plot_performance_metrics(lambdas, mse_scores, r2_scores):
    """
    Crea subgráficos para visualizar las métricas MSE y R² en función de λ.
    
    Parámetros:
    -----------
    lambdas : lista o array
        Valores de λ.
    mse_scores : lista o array
        Valores de MSE.
    r2_scores : lista o array
        Valores de R².
    
    Retorna:
    --------
    tuple
        (min_lambda, min_mse, max_lambda, max_r2) con los valores óptimos para cada métrica.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    min_lambda, min_mse = plot_metric_vs_lambda(ax1, lambdas, mse_scores, metric_name="MSE", marker=None)
    max_lambda, max_r2 = plot_metric_vs_lambda(ax2, lambdas, r2_scores, metric_name="R2", marker=None)
    
    plt.tight_layout()
    plt.suptitle('Métricas de rendimiento vs Intensidad de regularización', y=1.05, fontsize=16)
    plt.show()
    
    return min_lambda, min_mse, max_lambda, max_r2


def plot_cv_results(lambdas, cv_metrics_scores, optimal_lambda=None, min_cv_metrics=None, metric_name='mse', title=None, ax=None):
    """
    Grafica la variación de las métricas promedio de validación cruzada en función de λ utilizando seaborn.
    
    Parámetros:
    -----------
    lambdas : lista o array
        Valores de λ.
    cv_metrics_scores : dict o array
        Métricas promedio para cada λ. Si es dict, debe contener la métrica especificada.
    optimal_lambda : valor o dict, opcional
        Valor óptimo de λ que optimiza la métrica. Si es None, se calcula automáticamente.
    min_cv_metrics : valor o dict, opcional
        Valor óptimo de la métrica. Si es None, se calcula automáticamente.
    metric_name : str, opcional
        Nombre de la métrica a visualizar ('mse', 'r2', etc.). Por defecto 'mse'.
    title : str, opcional
        Título del gráfico.
    ax : matplotlib.axes.Axes, opcional
        Eje sobre el cual dibujar el gráfico.
    
    Retorna:
    --------
    matplotlib.axes.Axes
        Objeto de ejes con el gráfico generado.
    """
    if ax is None:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    sns.set_style("whitegrid")
    
    # determinar si cv_metrics_scores es un diccionario o un array
    if isinstance(cv_metrics_scores, dict):
        metric_name = metric_name.lower()
        if metric_name not in cv_metrics_scores:
            available_metrics = list(cv_metrics_scores.keys())
            raise ValueError(f"La métrica '{metric_name}' no está disponible. Métricas disponibles: {available_metrics}")
        metrics_values = cv_metrics_scores[metric_name]
    else:
        metrics_values = cv_metrics_scores
    
    # calcular el valor óptimo de λ y la métrica correspondiente si no se proporcionan
    if isinstance(optimal_lambda, dict) and metric_name in optimal_lambda:
        optimal_lambda_value = optimal_lambda[metric_name]
    else:
        optimal_lambda_value = optimal_lambda
        
    if isinstance(min_cv_metrics, dict) and metric_name in min_cv_metrics:
        optimal_metric_value = min_cv_metrics[metric_name]
    else:
        optimal_metric_value = min_cv_metrics
    
    is_minimize = any(term in metric_name.lower() for term in ['error', 'loss', 'mse', 'mae'])
    
    
    _, _, ax = _setup_plot_with_optimal_value(
        lambdas, 
        metrics_values, 
        metric_name.upper(), 
        is_minimize, 
        optimal_lambda_value, 
        optimal_metric_value, 
        ax
    )
    
    ax.set_xlabel('Intensidad de regularización (λ)', fontsize=12)
    ax.set_ylabel(f'{metric_name.upper()} - Validación Cruzada', fontsize=12)
    if title is None:
        title = f'Validación Cruzada: {metric_name.upper()} vs λ'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax