import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save_and_show_figure(fig, save_path, filename, show_figures, dpi=300):
    """
    Helper function to handle saving and displaying figures
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save/display
    save_path : str or None
        Directory path to save the figure
    filename : str
        Filename to use when saving
    show_figures : bool
        Indicates whether to display the figure
    dpi : int
        Resolution for the saved figure
    
    Returns:
    --------
    matplotlib.figure.Figure
        The input figure
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
    Creates comprehensive visualizations to evaluate a regression model.
    
    Parameters:
    -----------
    y_true : array-like
        True values of the target variable.
    y_pred : array-like
        Values predicted by the model.
    transform_func : callable, optional
        Function to transform the values (for example, np.exp).
    fig_size : tuple, optional
        Size of the figure (default (10, 5)).
    titles : dict, optional
        Custom titles for the plots. Keys: 'scatter', 'residuals', 'distribution', 'qq_plot'.
    save_path : str, optional
        Path where figures will be saved; if None, they won't be saved.
    show_figures : bool, optional
        Indicates whether to display the figures.
    fit_degree : int, optional
        Degree of polynomial fit (1=linear, 2=quadratic, etc.).
    
    Returns:
    --------
    dict
        Dictionary with the created figures.
    """
    def _initialize_plot_titles(titles):
        """
        Initializes and validates plot titles.
        
        Parameters:
        -----------
        titles : dict
            Custom titles.
        
        Returns:
        --------
        dict
            Dictionary with all necessary titles (filling in missing ones).
        """
        default_titles = {
            "scatter": "Precio real vs Precio predicho",
            "residuals": "Residuales vs Precio predicho",
            "distribution": "Distribución de residuales",
            "qq_plot": "Q-Q plot de residuales"
        }
        
        if not titles:
            return default_titles
        else:
            # add missing titles using default values
            for key in default_titles:
                if key not in titles:
                    titles[key] = default_titles[key]
            return titles

    titles = _initialize_plot_titles(titles)
    labels = {"actual": "Precio real", "predicted": "Precio predicho", "residuals": "Residuales"}
    
    if transform_func:
        y_true = transform_func(y_true)
        y_pred = transform_func(y_pred)
    
    residuals = y_true - y_pred  # calculate the residuals
    
    figures = {}
    figures["scatter"] = _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, save_path, show_figures, fit_degree)
    figures["residuals"] = _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, save_path, show_figures)
    figures["distribution"] = _create_distribution_plot(residuals, labels, titles, fig_size, save_path, show_figures)
    figures["qq_plot"] = _create_qq_plot(residuals, titles, fig_size, save_path, show_figures)
    
    return figures


def _create_scatter_plot(y_true, y_pred, labels, titles, fig_size, save_path, show_figures, fit_degree):
    """
    Creates a scatter plot between true and predicted values, and includes the perfect prediction line.
    
    Parameters:
    -----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    labels : dict
        Labels for the axes.
    titles : dict
        Plot titles.
    fig_size : tuple
        Figure size.
    save_path : str
        Path to save the figure.
    show_figures : bool
        Indicates whether to display the figure.
    fit_degree : int
        Degree of polynomial fit; if greater than 1, the fit is performed.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig_scatter = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.7, label="Data")
    
    x_data = np.array(y_true)
    y_data = np.array(y_pred)
    
    # determine min and max to plot the perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Predicción perfecta")
    
    # if a degree greater than 1 is specified, fit a polynomial to the data
    if fit_degree > 1:
        coeffs = np.polyfit(x_data, y_data, fit_degree)  # get polynomial coefficients
        p = np.poly1d(coeffs)  # create a polynomial function
        
        x_curve = np.linspace(min_val, max_val, 100)  # generate points for the curve
        y_curve = p(x_curve)  # evaluate the polynomial at the generated points
        
        plt.plot(x_curve, y_curve, 'r-', label=f"Ajuste polinómico (grado={fit_degree})")
    
    plt.xlabel(labels["actual"])
    plt.ylabel(labels["predicted"])
    plt.title(titles["scatter"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    return _save_and_show_figure(fig_scatter, save_path, "scatter_plot.png", show_figures)


def _create_residuals_plot(y_pred, residuals, labels, titles, fig_size, save_path, show_figures):
    """
    Creates a scatter plot of residuals versus predicted values.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted values.
    residuals : array-like
        Difference between true and predicted values.
    labels : dict
        Labels for the axes.
    titles : dict
        Plot titles.
    fig_size : tuple
        Figure size.
    save_path : str
        Path to save the figure.
    show_figures : bool
        Indicates whether to display the figure.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig_residuals = plt.figure(figsize=fig_size)
    
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7, label="Residuales")
    
    plt.axhline(y=0, color='r', linestyle='--', label="Residuales nulos")  # horizontal line at zero
    
    plt.xlabel(labels["predicted"])
    plt.ylabel(labels["residuals"])
    plt.title(titles["residuals"])
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    return _save_and_show_figure(fig_residuals, save_path, "residuals_plot.png", show_figures)


def _create_distribution_plot(residuals, labels, titles, fig_size, save_path, show_figures):
    """
    Creates a histogram of the residuals distribution and includes a theoretical normal distribution curve.
    
    Parameters:
    -----------
    residuals : array-like
        Regression residuals.
    labels : dict
        Labels for the axes.
    titles : dict
        Plot titles.
    fig_size : tuple
        Figure size.
    save_path : str
        Path to save the figure.
    show_figures : bool
        Indicates whether to display the figure.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
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
    # calculate the probability density function of a normal distribution
    normal_pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    
    plt.plot(x, normal_pdf, 'r-', label="Normal distribution")
    plt.legend()
    
    return _save_and_show_figure(fig_dist, save_path, "residuals_distribution.png", show_figures)


def _create_qq_plot(residuals, titles, fig_size, save_path, show_figures):
    """
    Creates a Q-Q plot to compare the quantiles of the residuals with the theoretical quantiles of a normal distribution.
    
    Parameters:
    -----------
    residuals : array-like
        Regression residuals.
    titles : dict
        Plot titles.
    fig_size : tuple
        Figure size.
    save_path : str
        Path to save the figure.
    show_figures : bool
        Indicates whether to display the figure.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The generated figure.
    """
    def erfinv(y):
        """
        Approximates the inverse error function using Winitzki's formula.
        
        Parameters:
        -----------
        y : float or array-like
            Value(s) for which the inverse is calculated.
        
        Returns:
        --------
        float or array-like
            Approximated value(s) of the inverse error function.
        """
        a = 0.147
        ln = np.log(1 - y**2)  # calculate the logarithm of (1 - y^2)
        term1 = 2/(np.pi * a) + ln/2
        return np.sign(y) * np.sqrt(np.sqrt(term1**2 - ln/a) - term1)  # calculation of the inverse
        
    fig_qq = plt.figure(figsize=fig_size)
    
    ordered_values = np.sort(residuals)  # sort the residuals
    n = len(residuals)
    
    # calculate percentages for each point (p = (i-0.5)/n)
    probs = (np.arange(1, n+1) - 0.5) / n
    
    # calculate theoretical quantiles using the erfinv function
    theoretical_quantiles = np.sqrt(2) * erfinv(2 * probs - 1)
    
    # fit a linear regression line between theoretical and observed quantiles
    slope, intercept = np.polyfit(theoretical_quantiles, ordered_values, 1)
    
    # calculate correlation between theoretical and observed quantiles
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
    Helper function to set up a plot with an optimal value marker
    
    Parameters:
    -----------
    lambdas : array-like
        lambda values
    metric_scores : array-like
        scores for each lambda
    metric_name : str
        name of the metric
    is_minimize : bool, optional
        whether the metric should be minimized (if None, inferred from metric name)
    optimal_lambda : float, optional
        pre-calculated optimal lambda value
    optimal_metric : float, optional
        pre-calculated optimal metric value
    ax : matplotlib.axes.Axes, optional
        axes to plot on
    
    Returns:
    --------
    tuple
        (optimal_lambda, optimal_metric, ax)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # plot the line
    sns.lineplot(x=lambdas, y=metric_scores, ax=ax, color='royalblue', label=metric_name)
    
    # determine if we're minimizing or maximizing
    if is_minimize is None:
        is_minimize = any(term in metric_name.lower() for term in ['error', 'loss', 'mse', 'mae'])
    
    # find the optimal value if not provided
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
    
    # mark the optimal point
    ax.scatter([optimal_lambda], [optimal_metric], color='red', s=100, zorder=5,
               label=f'{optimal_type} {metric_name}: {optimal_metric:.4f} (λ = {optimal_lambda:.4f})')
    ax.axvline(x=optimal_lambda, color='red', linestyle='--', alpha=0.7)
    
    ax.text(0.02, 0.98, f'Optimal λ: {optimal_lambda:.4f}\n{metric_name} {optimal_type.lower()}: {optimal_metric:.4f}',
            transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='top')
    
    return optimal_lambda, optimal_metric, ax


def plot_weights_vs_lambda(lambdas, weights, feature_names, custom_titles=None):
    """
    Plots weight values as a function of regularization λ using seaborn.
    
    Parameters:
    -----------
    lambdas : list or array
        λ values.
    weights : array-like
        Matrix where each column represents the weights of a feature for each λ.
    feature_names : list
        Feature names.
    custom_titles : dict, optional
        Custom titles with keys: 'title', 'xlabel', 'ylabel'.
    
    Returns:
    --------
    None
    """
    # default titles
    titles = {
        "title": "Ridge Regression: Weight Values vs Regularization Strength",
        "xlabel": "Regularization strength (λ)",
        "ylabel": "Weight Value"
    }
    
    # update with custom titles if provided
    if custom_titles:
        titles.update(custom_titles)

    plt.figure(figsize=(16, 6))
    sns.set_style("whitegrid")
    
    # create a DataFrame to facilitate use with seaborn
    data = pd.DataFrame(weights, columns=feature_names[:weights.shape[1]])
    data['lambda'] = lambdas
    
    # convert DataFrame to long format to plot multiple lines
    data_long = pd.melt(data, id_vars=['lambda'], var_name='feature', value_name='weight')
    
    ax = sns.lineplot(x='lambda', y='weight', hue='feature', data=data_long)
    
    plt.xlabel(titles["xlabel"], fontsize=12)
    plt.ylabel(titles["ylabel"], fontsize=12)
    plt.title(titles["title"], fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Features')
    
    plt.tight_layout()
    plt.show()


def plot_metric_vs_lambda(ax, lambdas, metric_scores, metric_name="MSE", marker='o'):
    """
    Plots a metric (for example, MSE or R²) as a function of λ.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis where the plot is drawn.
    lambdas : list or array
        λ values.
    metric_scores : list or array
        Metric values.
    metric_name : str, optional
        Metric name ("MSE" or "R2"). Default "MSE".
    marker : str, optional
        Marker for the optimal point.
    
    Returns:
    --------
    tuple
        (optimal_lambda, optimal_metric) with the optimal λ and metric.
    """
    is_minimize = metric_name.upper() != "R2"
    optimal_lambda, optimal_metric, _ = _setup_plot_with_optimal_value(
        lambdas, metric_scores, metric_name, is_minimize, ax=ax)
    
    ax.set_xlabel('Coeficiente de regularización (λ)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} vs λ')
    ax.grid(True)
    
    return optimal_lambda, optimal_metric


def plot_performance_metrics(lambdas, mse_scores, r2_scores, random_state=None):
    """
    Creates subplots to visualize MSE and R² metrics as a function of λ.
    
    Parameters:
    -----------
    lambdas : list or array
        λ values.
    mse_scores : list or array
        MSE values.
    r2_scores : list or array
        R² values.
    random_state : int, optional
        Random seed for reproducibility.
        This is passed to any function called that might require randomization.
    
    Returns:
    --------
    tuple
        (min_lambda, min_mse, max_lambda, max_r2) with the optimal values for each metric.
    """
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    min_lambda, min_mse = plot_metric_vs_lambda(ax1, lambdas, mse_scores, metric_name="MSE", marker=None)
    max_lambda, max_r2 = plot_metric_vs_lambda(ax2, lambdas, r2_scores, metric_name="R2", marker=None)
    
    plt.tight_layout()
    plt.suptitle('Métricas de rendimiento vs λ', y=1.05, fontsize=16)
    plt.show()
    
    return min_lambda, min_mse, max_lambda, max_r2


def plot_cv_results(lambdas, cv_metrics_scores, optimal_lambda=None, min_cv_metrics=None, metric_name='mse', title=None, ax=None):
    """
    Plots the variation of average cross-validation metrics as a function of λ using seaborn.
    
    Parameters:
    -----------
    lambdas : list or array
        λ values.
    cv_metrics_scores : dict or array
        Average metrics for each λ. If dict, must contain the specified metric.
    optimal_lambda : value or dict, optional
        Optimal λ value that optimizes the metric. If None, calculated automatically.
    min_cv_metrics : value or dict, optional
        Optimal metric value. If None, calculated automatically.
    metric_name : str, optional
        Name of the metric to visualize ('mse', 'r2', etc.). Default 'mse'.
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Axis on which to draw the plot.
    
    Returns:
    --------
    matplotlib.axes.Axes
        Axes object with the generated plot.
    """
    if ax is None:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
    sns.set_style("whitegrid")
    
    # determine if cv_metrics_scores is a dictionary or an array
    if isinstance(cv_metrics_scores, dict):
        metric_name = metric_name.lower()
        if metric_name not in cv_metrics_scores:
            available_metrics = list(cv_metrics_scores.keys())
            raise ValueError(f"The metric '{metric_name}' is not available. Available metrics: {available_metrics}")
        metrics_values = cv_metrics_scores[metric_name]
    else:
        metrics_values = cv_metrics_scores
    
    # calculate the optimal λ value and the corresponding metric if not provided
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
    
    ax.set_xlabel('Coeficiente de regularización (λ)', fontsize=12)
    ax.set_ylabel(f'{metric_name.upper()} - Cross Validation', fontsize=12)
    if title is None:
        title = f'Cross Validation: {metric_name.upper()} vs λ'
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax