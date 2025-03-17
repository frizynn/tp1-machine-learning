import pandas as pd
import numpy as np


def round_input(X, columnas):
    """
    Redondea a enteros las columnas especificadas de un DataFrame.
    """
    X_copy = X.copy()
    for col in columnas:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].round(0).astype(int)
    return X_copy


def get_nan_features(
    df: pd.DataFrame,
):
    nan_features = df.isnull().sum()
    nan_features = nan_features[nan_features > 0]
    nan_features = nan_features.to_dict()

    return nan_features


def split_by_nan_features(df, nan_features):

    exclusive_missing = []
    for (
        i,
        feat,
    ) in enumerate(nan_features):

        cond = df[feat].isnull()

        for (
            j,
            other,
        ) in enumerate(nan_features):
            if i != j:
                cond = cond & df[other].notnull()
        exclusive_missing.append(df[cond])

    cond_all_missing = df[nan_features].isnull().all(axis=1)
    all_missing = df[cond_all_missing]

    cond_all_present = df[nan_features].notnull().all(axis=1)
    all_present = df[cond_all_present]

    return (
        *exclusive_missing,
        all_missing,
        all_present,
    )


def split_test_train_with_label(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_target: bool = True,
    normalize: bool = False
) -> tuple:
    """
    Divide los datos en conjuntos de entrenamiento y prueba, con opción de normalización.
    
    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    test_size : float
        Proporción del conjunto de prueba
    random_state : int
        Semilla para reproducibilidad
    drop_target : bool
        Si debe eliminar la columna target de X
    normalize : bool
        Si debe normalizar los datos usando la media y std de X_train
        
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    if drop_target:
        if isinstance(y, pd.Series) and y.name in X.columns:
            X = X.drop(columns=[y.name])
        elif isinstance(y, pd.DataFrame):
            for col in y.columns:
                if col in X.columns:
                    X = X.drop(columns=[col])

    np.random.seed(random_state)
    n = len(X)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)
    X = X.iloc[idx]
    y = y.iloc[idx]
    
    X_train = X.iloc[:n_train]
    X_test = X.iloc[n_train:]
    y_train = y.iloc[:n_train]
    y_test = y.iloc[n_train:]
    
    if normalize:
        # Calcular media y desviación estándar solo con X_train
        train_mean = X_train.mean()
        train_std = X_train.std()
        
        # Evitar división por cero en columnas constantes
        train_std = train_std.replace(0, 1)
        
        # Normalizar tanto X_train como X_test usando estadísticas de X_train
        X_train = (X_train - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        # Guardar los parámetros de normalización como atributos del DataFrame
        X_train.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
        X_test.attrs['normalization'] = {'mean': train_mean, 'std': train_std}
    
    return X_train, X_test, y_train, y_test

def split_test_train_without_label(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state=42,
):
    np.random.seed(random_state)
    n = len(df)
    n_test = int(n * test_size)
    n_train = n - n_test
    idx = np.random.permutation(n)
    df = df.iloc[idx]
    train, test = (
        df.iloc[:n_train],
        df.iloc[n_train:],
    )
    return (
        train,
        test,
    )


from models.clustering.kmeans import KMeans
from typing import Dict, List, Union, Callable

def process_dataset(
    df: pd.DataFrame,
    kmeans_model: KMeans,
    feature_engineering_ops: List[Dict[str, Union[str, Callable]]] = None,
    features_to_impute: List[str] = ['age', 'rooms'],
    location_columns: List[str] = ['lat', 'lon'],
    save_path: str = None
) -> Dict:
    """
    Process dataset with clustering and feature engineering.
    """

    df = df.copy()
    
    if location_columns:
        location_data = df[location_columns].to_numpy()
        kmeans_model.fit(location_data)
        df['location_zone'] = kmeans_model.predict(location_data)
    

    if feature_engineering_ops is None:
        feature_engineering_ops = [
            {'name': 'area_per_room', 'operation': lambda df: df['area'] / df['rooms']},
            {'name': 'pool_house', 'operation': lambda df: df['has_pool'] * df['is_house']},
            {'name': 'house_area', 'operation': lambda df: df['area'] * df['is_house']}
        ]
    

    new_features_dict = {}
    feature_stats = {}
    
    for op in feature_engineering_ops:
        feature_name = op['name']
        operation = op['operation']
        
        if isinstance(operation, str):
            new_features_dict[feature_name] = df.eval(operation)
        else:
            new_features_dict[feature_name] = operation(df)
            
        feature_stats[feature_name] = {
            'mean': new_features_dict[feature_name].mean(),
            'std': new_features_dict[feature_name].std(),
            'min': new_features_dict[feature_name].min(),
            'max': new_features_dict[feature_name].max()
        }
    

    new_features_df = pd.DataFrame(new_features_dict)
    df = pd.concat([df, new_features_df], axis=1)
    
    if location_columns:
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
                'center': kmeans_model.cluster_centers_[zone]
            })
            
            zone_stats[f'zone_{zone}'] = stats
            
            for feature in features_to_impute:
                df.loc[df['location_zone'] == zone, feature] = df.loc[
                    df['location_zone'] == zone, feature
                ].fillna(stats[feature]['mean'])
        
        df_pos = df[location_columns + ['location_zone']] if location_columns else None
        df = df.drop(location_columns, axis=1) if location_columns else df

    if save_path is not None:
        df.to_csv(save_path, index=False)
    
    return {
        'df': df,
        'df_pos': df_pos if location_columns else None,
        'zone_stats': zone_stats if location_columns else None,
        'feature_stats': feature_stats
    }