import numpy as np
import pandas as pd

class Model:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        raise NotImplementedError
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed=42):
        np.random.seed(seed)
        n = len(X)
        n_test = int(n * test_size)
        n_train = n - n_test
        idx = np.random.permutation(n)
        X = X.iloc[idx]
        y = y.iloc[idx]
        X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
        y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
        return X_train, X_test, y_train, y_test

    def mse_score(self, X: pd.DataFrame, y, round=False):
        """
        Calcula el error cuadrático medio (MSE) entre la predicción y el target.
        Si y es un DataFrame (para predicción multivariante), se calcula el MSE promedio
        de todas las columnas.
        """
        y_pred = self.predict(X)
        if round:
            y_pred = np.round(y_pred)
        if isinstance(y, pd.DataFrame):
            # Convertir la predicción a DataFrame para alinear índices y columnas
            y_pred = pd.DataFrame(y_pred, index=y.index, columns=y.columns)
            mse = ((y - y_pred) ** 2).mean().mean()
        else:
            mse = ((y - y_pred) ** 2).mean()
        return mse

class LinearRegressor(Model):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Reiniciar índices para asegurar la alineación
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Agregar columna de unos para el intercepto
        X_with_intercept = pd.DataFrame({
            'intercept': np.ones(len(X))
        }).join(X)
        
        # Convertir a arrays de numpy para evitar problemas de alineación por índices
        X_np = X_with_intercept.values.astype(float)
        y_np = y.values.astype(float)
        
        # Calcular los coeficientes usando la fórmula de la regresión lineal
        coeffs = np.linalg.inv(X_np.T @ X_np) @ (X_np.T @ y_np)
        
        # Extraer intercepto y coeficientes
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        
        return self

    def predict(self, X: pd.DataFrame):
        # Asegurarse de que X tenga el formato numérico correcto
        X_np = X.values.astype(float)
        return self.intercept_ + X_np @ self.coef_
