import pandas as pd
import numpy as np
from models.clustering.kmeans import KMeans
from typing import Dict, List, Union, Callable, Tuple
from models.regression.base import Model


def evaluate_model(model, X_test, y_test, metrics=None, inv_transform_pred=None):
    """
    evalúa el rendimiento del modelo usando métricas específicas.
    
    parámetros:
    -----------
    model : Model
        instancia del modelo entrenado
    X_test : pd.DataFrame
        características de prueba
    y_test : pd.Series
        valores objetivo de prueba
    metrics : list, default=None
        lista de funciones de métrica para calcular
    inv_transform_pred : callable, default=None
        función para transformar predicciones a escala original
        
    retorna:
    --------
    dict
        diccionario con valores de las métricas
    """
    if metrics is None:
        metrics = [mse_score, r2_score]
    
    y_pred_test = model.predict(X_test)

    if inv_transform_pred is not None:
        y_pred_test = inv_transform_pred(y_pred_test)
        y_test = inv_transform_pred(y_test)

    results = {}
    for metric in metrics:
        results[metric.__name__.lower()] = metric(y_pred_test, y_test)
            
    return results


def preprocess_data(df, save_path=None):
    """
    Preprocesa datos inmobiliarios convirtiendo unidades de sqft a m2.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame con los datos a preprocesar.
    save_path : str, default=None
        Ruta para guardar los datos preprocesados.
        
    Retorna:
    --------
    pd.DataFrame
        DataFrame preprocesado.
    """
    # verificar que la columna 'area_units' exista
    if 'area_units' not in df.columns:
        raise ValueError("La columna 'area_units' no existe en el DataFrame.")
    # convierte sqft a m2
    df.loc[df['area_units'] == 'sqft', 'area'] = df['area'] * 0.092903  
    df['area_units'] = 'm2'  
    df = df.drop('area_units', axis=1)
    if save_path:
        df.to_csv(save_path, index=False)
    return df


def mse_score(y_pred, y_test, round=False):
    """
    Calcula el error cuadrático medio (MSE) entre la predicción y el target.
    Si y_test es un DataFrame (para predicción multivariante), se calcula el MSE promedio de todas las columnas.
    
    Parámetros:
    -----------
    y_pred : array-like
        Valores predichos.
    y_test : array-like o pd.DataFrame
        Valores reales.
    round : bool, opcional
        Si se debe redondear y_pred antes de calcular el error.
    
    Retorna:
    --------
    float
        Valor del MSE.
    """
    if round:
        y_pred = np.round(y_pred)  # redondea los valores predichos si se solicita
    if isinstance(y_test, pd.DataFrame):
        # convierte y_pred a DataFrame para cálculos multivariantes
        y_pred = pd.DataFrame(y_pred, index=y_test.index, columns=y_test.columns)
        mse = ((y_test - y_pred) ** 2).mean().mean()
    else:
        mse = ((y_test - y_pred) ** 2).mean()
    return mse


def r2_score(y_pred, y_test):
    """
    Calcula el coeficiente de determinación R^2 de la predicción.
    
    Parámetros:
    -----------
    y_pred : array-like
        Valores predichos.
    y_test : array-like
        Valores reales.
    
    Retorna:
    --------
    float
        Valor de R^2.
    """
    y_mean = y_test.mean()  # media de los valores reales
    ss_total = ((y_test - y_mean) ** 2).sum()  # suma total de cuadrados
    ss_res = ((y_test - y_pred) ** 2).sum()      # suma de cuadrados residuales
    r2 = 1 - ss_res / ss_total
    return r2


def round_input(X, columnas):
    """
    Redondea a enteros las columnas especificadas de un DataFrame.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        DataFrame de entrada.
    columnas : list
        Lista de nombres de columnas a redondear.
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con las columnas redondeadas.
    """
    X_copy = X.copy()
    for col in columnas:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].round(0).astype(int)  # redondea y convierte a entero
    return X_copy


def get_nan_features(df: pd.DataFrame):
    """
    Encuentra columnas con valores NaN y devuelve un diccionario con conteos.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame a analizar.
        
    Retorna:
    --------
    dict
        Diccionario con nombres de columnas como claves y conteos de NaN como valores.
    """
    nan_features = df.isnull().sum()  # cuenta los NaN por columna
    nan_features = nan_features[nan_features > 0]  # filtra solo columnas con al menos un NaN
    nan_features = nan_features.to_dict()
    return nan_features


def split_test_train(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_target: bool = True,
    transform_target: Callable = None,
    normalize: bool = False
) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba, con opción de normalización.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        Conjunto de características.
    y : pd.Series
        Variable objetivo.
    test_size : float
        Proporción del conjunto de prueba.
    random_state : int
        Semilla para reproducibilidad.
    drop_target : bool
        Indica si se debe eliminar la columna objetivo de X.
    transform_target : Callable, opcional
        Función para transformar la variable objetivo.
    normalize : bool
        Indica si se deben normalizar los datos usando la media y std de X_train.
        
    Retorna:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if drop_target:
        # elimina la columna objetivo de X si está presente
        if isinstance(y, pd.Series) and y.name in X.columns:
            X = X.drop(columns=[y.name])
        elif isinstance(y, pd.DataFrame):
            for col in y.columns:
                if col in X.columns:
                    X = X.drop(columns=[col])
    
    np.random.seed(random_state)  # fija la semilla para reproducibilidad
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)  # genera una permutación aleatoria de índices
    X = X.iloc[idx]
    y = y.iloc[idx]
    
    if transform_target:
        y = transform_target(y)
    
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    
    if normalize:
        # calcula media y desviación estándar a partir de X_train
        train_mean = X_train.mean()
        train_std = X_train.std()
        train_std = train_std.replace(0, 1)  # evita división por cero en características constantes
        # normaliza X_train y X_test usando estadísticas de entrenamiento
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        # almacena los parámetros de normalización en los atributos del DataFrame
        X_train.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
        X_test.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
    
    return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    target_column,
    df=None,
    data_path=None,
    feature_columns=None,
    transform_target=None
):
    """
    Carga datos desde un DataFrame o archivo CSV y prepara las características y la variable objetivo.
    
    Parámetros:
    -----------
    target_column : str
        Nombre de la columna de la variable objetivo.
    df : pd.DataFrame, default=None
        DataFrame con datos.
    data_path : str, default=None
        Ruta al archivo CSV si no se proporciona df.
    feature_columns : list, default=None
        Lista de columnas a utilizar como características. Si es None, se usan todas excepto la columna objetivo.
    transform_target : callable, default=None
        Función para transformar la variable objetivo.
        
    Retorna:
    --------
    tuple
        (X, y, feature_columns)
    """
    # carga el DataFrame desde archivo si no se proporciona
    df = pd.read_csv(data_path) if df is None else df.copy()
    
    if transform_target:
        df[target_column] = transform_target(df[target_column])
    
    if feature_columns is None:
        # selecciona todas las columnas excepto la columna objetivo
        feature_columns = [col for col in df.columns if col != target_column]
    
    X = df[feature_columns]
    y = df[target_column]
    
    return X, y, feature_columns


def _calculate_normalization_params(data: pd.DataFrame) -> Dict:
    """
    Calcula parámetros de normalización (media y desviación estándar) a partir de los datos.
    
    Parámetros:
    -----------
    data : pd.DataFrame
        Datos para calcular parámetros de normalización.
        
    Retorna:
    --------
    dict
        Diccionario con claves 'mean' y 'std'.
    """
    binary_cols = data.apply(lambda x: set(x.unique()) <= {0, 1})
    # selecciona solo las columnas que no son binarias
    non_binary_data = data.loc[:, ~binary_cols]
    
    # inicializa parámetros por defecto para todas las columnas
    params = {
        'mean': pd.Series(0.0, index=data.columns, dtype=float),
        'std': pd.Series(1.0, index=data.columns, dtype=float)
    }
    
    # actualiza parámetros solo para columnas no binarias
    if not non_binary_data.empty:
        means = non_binary_data.mean()
        stds = non_binary_data.std().replace(0, 1.0)  # evita división por cero
        for col in non_binary_data.columns:
            params['mean'][col] = means[col]
            params['std'][col] = stds[col]
    
    return params


def _apply_normalization(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Aplica normalización a los datos usando los parámetros dados.
    
    Parámetros:
    -----------
    data : pd.DataFrame
        Datos a normalizar.
    params : dict
        Diccionario con claves 'mean' y 'std'.
        
    Retorna:
    --------
    pd.DataFrame
        Datos normalizados.
    """
    # normaliza restando la media y dividiendo por la desviación estándar
    return (data - params['mean']) / params['std']


def normalize_data(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple:
    """
    Normaliza características usando estadísticas calculadas de los datos de entrenamiento.
    
    Parámetros:
    -----------
    X_train : pd.DataFrame
        Datos de entrenamiento.
    X_test : pd.DataFrame
        Datos de prueba.
        
    Retorna:
    --------
    tuple
        (X_train_normalizado, X_test_normalizado, parámetros de normalización)
    """
    params = _calculate_normalization_params(X_train)
    X_train_normalized = _apply_normalization(X_train, params)
    X_test_normalized = _apply_normalization(X_test, params)
    return X_train_normalized, X_test_normalized, params


def _handle_feature_engineering(df: pd.DataFrame, operations: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Aplica operaciones de ingeniería de características y recopila estadísticas de las nuevas características.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        DataFrame sobre el cual aplicar las operaciones.
    operations : List[Dict]
        Lista de diccionarios con claves 'name' y 'operation'.
        
    Retorna:
    --------
    Tuple
        (new_features_df, feature_stats) donde new_features_df es un DataFrame con las nuevas características y feature_stats es un diccionario con estadísticas.
    """
    new_features = {}
    feature_stats = {}
    
    for op in operations:
        feature_name = op['name']
        operation = op['operation']
        # aplica la operación: si es callable se ejecuta, de lo contrario se evalúa la expresión
        new_features[feature_name] = operation(df) if callable(operation) else df.eval(operation)
        # recopila estadísticas básicas de la nueva característica
        feature_stats[feature_name] = {
            'mean': new_features[feature_name].mean(),
            'std': new_features[feature_name].std(),
            'min': new_features[feature_name].min(),
            'max': new_features[feature_name].max()
        }
    
    return pd.DataFrame(new_features), feature_stats


def process_dataset(
    df: pd.DataFrame,
    kmeans_model: KMeans,
    feature_engineering_ops: List[Dict[str, Union[str, Callable]]] = None,
    features_to_impute: List[str] = ['age', 'rooms'],
    location_columns: List[str] = ['lat', 'lon'],
    impute_by_zone: bool = False,
    save_path: str = None,
    create_zone_interactions: bool = False,
    zone_interaction_features: List[str] = ['area', 'rooms', 'age', 'has_pool', 'is_house']
) -> Dict:
    """
    Procesa un conjunto de datos aplicando agrupamiento (clustering) y feature engineering.
    
    Parámetros:
    -----------
    df : pd.DataFrame
        Datos a procesar.
    kmeans_model : KMeans
        Modelo KMeans para agrupar ubicaciones.
    feature_engineering_ops : List[Dict], opcional
        Operaciones para ingeniería de características.
    features_to_impute : List[str]
        Características que requieren imputación de valores faltantes.
    location_columns : List[str]
        Columnas que contienen datos de ubicación.
    impute_by_zone : bool
        Si es True, imputa valores faltantes usando la media por zona.
    save_path : str, opcional
        Ruta para guardar el DataFrame procesado.
    create_zone_interactions : bool
        Si es True, crea características de interacción entre la zona y otras características.
    zone_interaction_features : List[str]
        Características a interactuar con la zona.
        
    Retorna:
    --------
    Dict
        Diccionario con:
            - 'df': DataFrame procesado sin columnas de ubicación.
            - 'df_pos': DataFrame con columnas de ubicación y zona.
            - 'zone_stats': Estadísticas por zona (si se usan ubicaciones).
            - 'feature_stats': Estadísticas de las nuevas características.
    """
    df = df.copy()
    
    if location_columns:
        # agrupa ubicaciones usando kmeans
        location_data = df[location_columns].to_numpy()  
        kmeans_model.fit(location_data)
        df['location_zone'] = kmeans_model.predict(location_data)
        
        # calcula estadísticas por zona para posible imputación
        zone_stats = {}
        for zone in range(kmeans_model.n_clusters):
            df_zone = df[df['location_zone'] == zone]
            stats = {feat: {
                'mean': df_zone[feat].mean(),
                'median': df_zone[feat].median(),
                'std': df_zone[feat].std(),
                'missing': df_zone[feat].isna().sum()
            } for feat in features_to_impute}
            stats.update({
                'size': len(df_zone),
                'percentage': len(df_zone) / len(df) * 100,
                'center': kmeans_model.cluster_centers_[zone]  # centroide de la zona
            })
            zone_stats[f'zone_{zone}'] = stats
            
            # imputa valores faltantes por zona si se solicita
            if impute_by_zone:
                for feature in features_to_impute:
                    df.loc[df['location_zone'] == zone, feature] = df.loc[
                        df['location_zone'] == zone, feature
                    ].fillna(stats[feature]['mean'])
    else:
        zone_stats = None
    
    # imputación global si no se imputa por zona
    if not impute_by_zone and features_to_impute:
        for feature in features_to_impute:
            global_mean = df[feature].mean()
            df[feature] = df[feature].fillna(global_mean)
    
    # define operaciones por defecto si no se proporcionan
    if feature_engineering_ops is None:
        feature_engineering_ops = [
            {'name': 'pool_house', 'operation': lambda df: df['has_pool'] * df['is_house']},
            {'name': 'house_area', 'operation': lambda df: df['area'] * df['is_house']},
            {'name': 'dist_to_cluster_center', 'operation': lambda df: 
                np.array([
                    # calcula la distancia euclidiana desde la ubicación de cada registro hasta el centroide de su zona
                    np.linalg.norm(
                        df.loc[i, location_columns].values - kmeans_model.cluster_centers_[df.loc[i, 'location_zone'].astype(int)]
                    )
                    for i in df.index
                ])
            }
        ]
    
    new_features_df, feature_stats = _handle_feature_engineering(df, feature_engineering_ops)
    df = pd.concat([df, new_features_df], axis=1)
    
    # crea características de interacción entre zona y otras características si se solicita
    if create_zone_interactions and 'location_zone' in df.columns:
        zone_interactions = {}
        for feature in zone_interaction_features:
            if feature in df.columns:
                for zone in range(kmeans_model.n_clusters):
                    interaction_name = f'{feature}_zone_{zone}'
                    # crea variable dummy: multiplicación de la característica por la condición de pertenencia a la zona
                    zone_interactions[interaction_name] = df[feature] * (df['location_zone'] == zone)
        if zone_interactions:
            zone_interactions_df = pd.DataFrame(zone_interactions)
            df = pd.concat([df, zone_interactions_df], axis=1)
            # actualiza las estadísticas de las nuevas interacciones
            for col in zone_interactions:
                feature_stats[col] = {
                    'mean': zone_interactions[col].mean(),
                    'std': zone_interactions[col].std(),
                    'min': zone_interactions[col].min(),
                    'max': zone_interactions[col].max()
                }
    
    # extrae datos de posición si están disponibles
    df_pos = df[location_columns + ['location_zone']] if location_columns else None
    # elimina las columnas de ubicación del DataFrame final
    df = df.drop(location_columns, axis=1) if location_columns else df

    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return {
        'df': df,
        'df_pos': df_pos if location_columns else None,
        'zone_stats': zone_stats if location_columns else None,
        'feature_stats': feature_stats
    }


def compare_feature_impact(model_results_dict, property_dfs, feature_name='has_pool'):
    """
    Compara el impacto de una característica específica entre diferentes modelos.
    
    Parámetros:
    -----------
    model_results_dict : dict
        Diccionario de resultados de modelos o un solo diccionario de resultados.
    property_dfs : dict
        Diccionario de DataFrames o un solo DataFrame.
    feature_name : str
        Nombre de la característica a analizar.
        
    Retorna:
    --------
    dict
        Diccionario con resultados de análisis de impacto para cada modelo.
    """
    # si se proporciona un solo modelo, encapsularlo en un diccionario
    if not isinstance(model_results_dict, dict) or (isinstance(model_results_dict, dict) and 'model' in model_results_dict):
        model_results_dict = {"Modelo": model_results_dict}
        property_dfs = {"Modelo": property_dfs}
    
    if not isinstance(property_dfs, dict):
        raise ValueError("property_dfs must be a dictionary")
    
    if set(model_results_dict.keys()) != set(property_dfs.keys()):
        raise ValueError("model_results_dict and property_dfs must have the same keys")
    
    impacts = {}
    for model_name, model_results in model_results_dict.items():
        # intenta obtener el diccionario de coeficientes del modelo
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
        avg_property_price = df["price"].mean()  # precio promedio de propiedades
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


def cross_validate_lambda(X, y, lambdas, model_class: Model, n_splits=5, 
                          method='pseudo_inverse', regularization='l2', 
                          normalize=True, random_state=None,
                          variable: str = 'penalty',
                          transform_target=None,
                          metrics=None,
                          inv_transform_pred=None):
    """
    Realiza validación cruzada para diferentes valores de lambda (o grado) y devuelve métricas promedio.
    
    Parámetros:
    -----------
    X : pd.DataFrame
        Matriz de características.
    y : pd.Series
        Variable objetivo.
    lambdas : iterable
        Valores de lambda (o grado) a evaluar.
    model_class : Model
        Clase del modelo de regresión a usar.
    n_splits : int
        Número de particiones para validación cruzada.
    method : str
        Método de ajuste para el modelo.
    regularization : str
        Tipo de regularización ('l1', 'l2').
    normalize : bool
        Si se deben normalizar las características.
    random_state : int
        Semilla para reproducibilidad.
    variable : str
        Parámetro a optimizar ('penalty' o 'degree').
    transform_target : callable, opcional
        Función para transformar la variable objetivo.
    metrics : list, opcional
        Lista de funciones de métricas a evaluar.
    inv_transform_pred : callable, opcional
        Función para invertir la transformación de la variable objetivo.
        
    Retorna:
    --------
    tuple
        (optimal_lambda, min_cv_metrics, cv_metrics_scores)
    """
    if metrics is None:
        metrics = [mse_score, r2_score]

    if transform_target and inv_transform_pred is None:
        raise ValueError("transform_target y inv_transform_pred no pueden ser None al mismo tiempo. Si se usa transform_target, se debe usar inv_transform_pred para invertir la transformación para calcular las métricas.")
    
    def numpy_kfold(n_samples, n_splits, random_state=None):
        """implementación personalizada de validación cruzada k-fold"""
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)  # mezcla los índices de forma aleatoria
        fold_size = n_samples // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < n_splits - 1 else n_samples
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            yield train_indices, val_indices

    n_samples = len(X)
    folds = list(numpy_kfold(n_samples, n_splits, random_state))
    
    # inicializa diccionario para almacenar las métricas de validación cruzada
    cv_metrics_scores = {metric.__name__.lower(): [] for metric in metrics}
    
    # iterar sobre cada valor de lambda (o grado)
    for lambda_val in lambdas:
        # almacena las métricas de cada partición para el lambda actual
        fold_metrics_scores = {metric.__name__.lower(): [] for metric in metrics}
        for train_idx, val_idx in folds:
            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()
            
            if transform_target:
                y_train = transform_target(y_train)
                y_val = transform_target(y_val)
            
            if normalize:
                # normaliza usando estadísticas calculadas de X_train
                mean = X_train.mean()
                std = X_train.std().replace(0, 1e-8)  # evita división por cero
                X_train = (X_train - mean) / std
                X_val = (X_val - mean) / std
            
            model = model_class()
            if variable == 'degree':
                # para regresión polinómica: cambia el grado y ajusta el modelo
                model.change_degree(lambda_val)
                model.fit(X_train, y_train, regularization=regularization)
            elif variable == 'penalty':
                # para modelos regularizados: ajusta usando el parámetro de penalización
                model.fit(X_train, y_train, method=method, alpha=lambda_val, regularization=regularization)
            else:
                model.fit(X_train, y_train, regularization=regularization)
            
            score = evaluate_model(model, X_val, y_val, metrics, inv_transform_pred)
           
            for key, value in score.items():
                fold_metrics_scores[key].append(value)

        # promedia las métricas sobre todos los folds para el lambda actual
        for metric in metrics:
            key = metric.__name__.lower()
            mean_score = np.mean(fold_metrics_scores[key])
            cv_metrics_scores[key].append(mean_score)
    
    # convierte las listas de métricas a arrays de NumPy
    cv_metrics_scores = {key: np.array(scores) for key, scores in cv_metrics_scores.items()}
    
    # determina el índice óptimo para cada métrica (mínimo para errores, máximo para métricas como r2)
    optimal_idx = {}
    for key, scores in cv_metrics_scores.items():
        if any(term in key for term in ['error', 'loss', 'mse', 'mae']):
            optimal_idx[key] = np.argmin(scores)
        else:
            optimal_idx[key] = np.argmax(scores)
    
    optimal_lambda = {key: lambdas[optimal_idx[key]] for key in cv_metrics_scores}
    min_cv_metrics = {key: cv_metrics_scores[key][optimal_idx[key]] for key in cv_metrics_scores}
    
    return optimal_lambda, min_cv_metrics, cv_metrics_scores