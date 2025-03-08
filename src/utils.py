import numpy as np
import pandas as pd
from models import LinearRegressor,Model


def round_input(X, columnas):
    """
    Redondea a enteros las columnas especificadas de un DataFrame.
    """
    X_copy = X.copy()
    for col in columnas:
        if col in X_copy.columns:
            X_copy[col] = X_copy[col].round(0).astype(int)
    return X_copy

def get_nan_features(df: pd.DataFrame):
    nan_features = df.isnull().sum()
    nan_features = nan_features[nan_features > 0]
    nan_features = nan_features.index.tolist()


    return nan_features

def split_by_nan_features(df, nan_features):
    
    exclusive_missing = []
    for i, feat in enumerate(nan_features):

        cond = df[feat].isnull()

        for j, other in enumerate(nan_features):
            if i != j:
                cond = cond & df[other].notnull()
        exclusive_missing.append(df[cond])
    
    cond_all_missing = df[nan_features].isnull().all(axis=1)
    all_missing = df[cond_all_missing]
    
    cond_all_present = df[nan_features].notnull().all(axis=1)
    all_present = df[cond_all_present]
    
 
    return (*exclusive_missing, all_missing, all_present)

def train_model_for_feature(model_class: Model, X, y, seed=42):
    model = model_class()

    X_train, X_test, y_train, y_test = model.split_data(X, y, test_size=0.2, seed=seed)

    model.fit(X_train, y_train)

    return model, X_test, y_test