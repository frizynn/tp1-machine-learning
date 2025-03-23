import numpy as np


from .data import (
    split_test_train,
    load_and_prepare_data,
    normalize_data,
    mse_score,
    r2_score,
    evaluate_model
)

def prepare_and_evaluate_test_data(
    df_test, 
    model, 
    normalization_params=None, 
    target_column='price',
    feature_columns=None,
    transform_target_func=None,
    inv_transform_pred=None,
    metrics=None,
    print_metrics=False,
    round_digits=4
):
    """
    prepara datos de prueba normalizando características y transformando la variable objetivo, 
    luego evalúa un modelo.
    
    parámetros:
    -----------
    df_test : pd.DataFrame
        datos de prueba con características y variable objetivo
    model : object
        modelo entrenado para evaluar
    normalization_params : dict, default=None
        diccionario con parámetros de normalización 'mean' y 'std' del entrenamiento
        si es None, no se aplica normalización
    target_column : str, default='price'
        nombre de la columna objetivo en df_test
    feature_columns : list, default=None
        lista de columnas de características a usar. si es None, usa todas las columnas excepto la objetivo
    transform_target_func : callable, default=None
        función para transformar la variable objetivo (ej., np.log)
        si es None, no se aplica transformación
    inv_transform_pred : callable, default=None
        función para transformar predicciones a escala original (ej., np.exp)
    metrics : list, default=None
        lista de funciones de métrica para calcular
    print_metrics : bool, default=False
        indica si se imprimen los resultados de las métricas formateados
    round_digits : int, default=4
        número de decimales para redondear métricas al imprimir
        
    retorna:
    --------
    tuple
        (metrics_dict, X_test, y_test, y_test_transformed)
        - metrics_dict: diccionario con métricas de evaluación
        - X_test: matriz de características procesada
        - y_test: valores originales de la variable objetivo
        - y_test_transformed: valores transformados de la variable objetivo (si se aplicó transformación)
    """
    if feature_columns is None:
        feature_columns = [col for col in df_test.columns if col != target_column]
    
    X_test = df_test[feature_columns].copy() if feature_columns else df_test.drop(target_column, axis=1)
    y_test = df_test[target_column].copy()
    
    if normalization_params is not None:
        for col in X_test.columns:
            if col in normalization_params['mean'] and col in normalization_params['std']:
                X_test[col] = (X_test[col] - normalization_params['mean'][col]) / normalization_params['std'][col]
    
    if transform_target_func is not None:
        y_test_transformed = transform_target_func(y_test)
    else:
        y_test_transformed = y_test
    
    metrics_dict = evaluate_model(model, X_test, y_test_transformed, metrics, inv_transform_pred)
    
    if print_metrics:
        for metric_name, value in metrics_dict.items():
            print(f"{metric_name.upper()}: {value:.{round_digits}f}")
    
    return metrics_dict, X_test, y_test, y_test_transformed

def train_and_evaluate_model(
    target_column, 
    df=None,
    data_path=None, 
    feature_columns=None, 
    test_size=0.2, 
    random_state=42, 
    model_class=None,
    normalize_features=True,
    transform_target=None,
    inv_transform_pred=None,
    fit_params=None,
    metrics=None,
    verbose=True
):
    """
    Carga datos, preprocesa, entrena modelo y evalúa rendimiento.
    
    Parámetros:
    -----------
    target_column : str
        Nombre de la columna objetivo
    df : pd.DataFrame, default=None
        DataFrame con los datos
    data_path : str, default=None
        Ruta al archivo CSV si no se proporciona df
    feature_columns : list, default=None
        Lista de columnas de características a usar. Si es None, usa todas las columnas excepto la objetivo
    test_size : float, default=0.2
        Proporción de datos a usar para pruebas
    random_state : int, default=42
        Semilla aleatoria para reproducibilidad
    model_class : class, default=None
        Clase del modelo a usar (debe tener métodos fit y predict)
    normalize_features : bool, default=True
        Indica si se estandarizan las características
    transform_target : callable, default=None
        Función para transformar la variable objetivo
    inv_transform_pred : callable, default=None
        Función para transformar predicciones a escala original
    fit_params : dict, default=None
        Parámetros para pasar al método fit del modelo
    metrics : list, default=None
        Lista de funciones de métrica para calcular
    verbose : bool, default=True
        Indica si se imprimen los resultados
        
    Retorna:
    --------
    dict
        Diccionario con modelo, divisiones de datos y métricas de evaluación
    """
    
    # parámetros por defecto
    if fit_params is None:
        fit_params = {'method': 'gradient_descent', 'learning_rate': 0.01, 'epochs': 1000}
    
    if metrics is None:
        metrics = [mse_score, r2_score]
        
    if model_class is None:
       raise Exception("Es necesario especificar el modelo")
    
    if transform_target and not inv_transform_pred:
        raise Exception("Es necesario especificar la función de inv_transform_pred")
    
    # cargar y preparar datos
    X, y, feature_columns = load_and_prepare_data(
        target_column, df, data_path, feature_columns, transform_target
    )
    
    # dividir datos
    X_train, X_test, y_train, y_test = split_test_train(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # normalizar características si se solicita
    if normalize_features:
        X_train, X_test, normalization_params = normalize_data(X_train, X_test)
    
    # entrenar modelo
    model = model_class()
    model.fit(X_train, y_train, **fit_params)
    
    # predecir en conjunto de prueba
    y_pred_test = model.predict(X_test)
    
    # evaluar modelo
    metrics_results = evaluate_model(model, X_test, y_test, metrics, inv_transform_pred, y_pred_test)

    if verbose:
        model.print_coefficients()
    
    # preparar resultados
    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred_test': y_pred_test,
        'feature_columns': feature_columns,
        **metrics_results
    }
    
    if normalize_features:
        results['normalization_params'] = normalization_params
    
    return results

def get_weights_and_metrics(X, y, lambdas, model_class, test_size=0.2, random_state=42, normalize=True, regularization='l2',
                            method='pseudo_inverse', inv_transform_pred=None, metrics=None):
    """
    Obtiene los pesos del modelo y métricas de rendimiento para diferentes valores de lambda.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        Matriz de características
    y : pd.Series
        Variable objetivo
    lambdas : array-like
        Valores del parámetro de regularización a probar
    model_class : class
        Clase del modelo a utilizar
    test_size : float, default=0.2
        Proporción de datos a utilizar para pruebas
    random_state : int, default=42
        Semilla aleatoria para reproducibilidad
    normalize : bool, default=True
        Indica si se normalizan las características
    regularization : str, default='l2'
        Tipo de regularización a utilizar (l1 o l2)
    method : str, default='pseudo_inverse'
        Método a utilizar para resolver el sistema lineal
    inv_transform_pred : callable, default=None
        Función para transformar las predicciones a la escala original
    metrics : list, default=None
        Lista de funciones de métricas a calcular
        
    Retorna:
    --------
    tuple
        (weights, mse_scores, r2_scores) arrays
    """
    if metrics is None:
        metrics = [mse_score, r2_score]

    X_train, X_test, y_train, y_test = split_test_train(X, y, test_size=test_size, random_state=random_state, normalize=normalize)

    weights = []
    metric_scores = {metric.__name__.lower(): [] for metric in metrics}

    for lambda_ in lambdas:
        model = model_class()
        model.fit(X_train, y_train, method=method, alpha=lambda_, regularization=regularization)

        coefs = model.get_weights()
        weights.append(coefs)

        score = evaluate_model(model, X_test, y_test, metrics, inv_transform_pred)
        
        for key, value in score.items():
            metric_scores[key].append(value)

    weights = np.array(weights)
    
    mse_scores = metric_scores.get('mse_score', metric_scores.get('mse', []))
    r2_scores = metric_scores.get('r2_score', metric_scores.get('r2', []))
    
    return weights, mse_scores, r2_scores