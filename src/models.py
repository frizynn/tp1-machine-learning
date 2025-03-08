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
            y_pred = pd.DataFrame(y_pred, index=y.index, columns=y.columns)
            mse = ((y - y_pred) ** 2).mean().mean()
        else:
            mse = ((y - y_pred) ** 2).mean()
        return mse
    
    @staticmethod
    def _build_design_matrix(X: pd.DataFrame, degree: int = 1) -> pd.DataFrame:

        X = X.reset_index(drop=True)
        if degree == 1:
            intercept = pd.DataFrame({'intercept': np.ones(len(X))})

            return pd.concat([intercept, X.reset_index(drop=True)], axis=1)
        else:
            X_poly = pd.DataFrame({'intercept': np.ones(len(X))})
            for i in range(1, degree + 1):
                for col in X.columns:
                    X_poly[f"{col}^{i}"] = X[col] ** i
            return X_poly


class LinearRegressor(Model):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, degree=1)  # misma función para construir la matriz
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)
        

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        return self

    def predict(self, X: pd.DataFrame):
        X_design = self._build_design_matrix(X, degree=1)
        X_np = X_design.values.astype(float)
        return X_np @ np.concatenate(([self.intercept_], self.coef_))

class PolinomialRegressor(Model):
    @classmethod
    def change_degree(cls, degree):
        cls.default_degree = degree
        return cls

    def __init__(self, degree=None):
        if degree is None:
            degree = getattr(self.__class__, "default_degree", 2)
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)
        
        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self.feature_names = X.columns  
        return self

    def predict(self, X: pd.DataFrame):
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        return X_np @ np.concatenate(([self.intercept_], self.coef_))
