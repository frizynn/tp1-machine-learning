import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict


class FitMethod(Enum):
    """Enum para los métodos de entrenamiento disponibles."""
    PSEUDO_INVERSE = "pseudo_inverse"
    GRADIENT_DESCENT = "gradient_descent"


class Model:
    def fit(self, X: pd.DataFrame, y: pd.Series):
        raise NotImplementedError

    def predict(self, X: pd.DataFrame):
        raise NotImplementedError
    
   
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
        
    def print_coefficients(self, format_precision: int = 4) -> None:
        """
        Imprime los coeficientes del modelo con los nombres de sus variables.
        
        Parameters:
        -----------
        format_precision : int
            Número de decimales a mostrar
        """
        if self.coef_ is None or self.feature_names is None:
            print("El modelo no ha sido entrenado aún.")
            return
            
        print(f"Método: {self._training_info.get('method', 'desconocido')}")
        print(f"Intercept: {self.intercept_:.{format_precision}f}\n")
        print("Coeficientes:")
        print("-" * 30)
        
        for name, coef in zip(self.feature_names, self.coef_):
            print(f"{name:<15} | {coef:+.{format_precision}f}")
            
        if 'final_mse' in self._training_info:
            print(f"\nMSE final: {self._training_info['final_mse']:.{format_precision}f}")
            
        if self._training_info.get('method') == 'gradient_descent':
            print(f"Convergencia: {'Sí' if self._training_info['converged'] else 'No'}")
            print(f"Iteraciones: {self._training_info['final_epoch']}/{self._training_info['epochs']}")

    def get_training_info(self) -> Dict:
        """
        Obtiene información sobre el entrenamiento del modelo.
        
        Returns:
        --------
        Dict
            Diccionario con información sobre el entrenamiento
        """
        return self._training_info.copy()


class LinearRegressor(Model):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.feature_names = None
        self._training_info = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, method: Union[str, FitMethod] = "pseudo_inverse",
            gradient_params: Optional[Dict] = None) -> 'LinearRegressor':
        """
        Entrena el modelo usando el método especificado
        """
        if isinstance(method, str):
            method = method.lower()
            if method == "pseudo_inverse":
                method = FitMethod.PSEUDO_INVERSE
            elif method == "gradient_descent":
                method = FitMethod.GRADIENT_DESCENT
            else:
                raise ValueError(f"Método no reconocido: {method}")

        self.feature_names = X.columns
        
        if method == FitMethod.PSEUDO_INVERSE:
            self._fit_pseudo_inverse(X, y)
        elif method == FitMethod.GRADIENT_DESCENT:
            if gradient_params is None:
                gradient_params = {}
            X_design = self._build_design_matrix(X, degree=1)
            self._fit_gradient_descent(X_design.values, y.values, **gradient_params)
            
        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, degree=1)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        return self
    
    def _fit_gradient_descent(self, 
                             X_np: np.ndarray, 
                             y_np: np.ndarray, 
                             lr: float = 0.01, 
                             epochs: int = 1000,
                             tolerance: float = 1e-6) -> None:
        m, n = X_np.shape
        coeffs = np.zeros(n)
        prev_mse = float('inf')
        
        history = {
            'mse': [],
            'iterations': 0
        }
        
        for epoch in range(epochs):
            y_pred = X_np @ coeffs
            error = y_pred - y_np
            mse = np.mean(error**2)
            history['mse'].append(mse)
            
            if abs(prev_mse - mse) < tolerance:
                break
                
            prev_mse = mse
            gradients = (2/m) * (X_np.T @ error)
            coeffs -= lr * gradients
        
        history['iterations'] = epoch + 1
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self._training_info = {
            'method': 'gradient_descent',
            'lr': lr,
            'epochs': epochs,
            'final_epoch': epoch + 1,
            'final_mse': mse,
            'converged': epoch < epochs - 1,
            'history': history
        }

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
        self.feature_names = None
        self._training_info = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, method: Union[str, FitMethod] = "pseudo_inverse",
            gradient_params: Optional[Dict] = None) -> 'PolinomialRegressor':
        """
        Entrena el modelo usando el método especificado
        """
        if isinstance(method, str):
            method = method.lower()
            if method == "pseudo_inverse":
                method = FitMethod.PSEUDO_INVERSE
            elif method == "gradient_descent":
                method = FitMethod.GRADIENT_DESCENT
            else:
                raise ValueError(f"Método no reconocido: {method}")
        
        self.feature_names = X.columns
        
        if method == FitMethod.PSEUDO_INVERSE:
            self._fit_pseudo_inverse(X, y)
        elif method == FitMethod.GRADIENT_DESCENT:
            if gradient_params is None:
                gradient_params = {}
            X_design = self._build_design_matrix(X, self.degree)
            self._fit_gradient_descent(X_design.values, y.values, **gradient_params)
            
        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self._training_info = {
            'method': 'pseudo_inverse',
            'final_mse': self.mse_score(X, y)
        }
        return self
    
    def _fit_gradient_descent(self, 
                            X_np: np.ndarray, 
                            y_np: np.ndarray, 
                            lr: float = 0.01, 
                            epochs: int = 1000,
                            tolerance: float = 1e-6) -> None:
        m, n = X_np.shape
        coeffs = np.zeros(n)
        prev_mse = float('inf')
        
        history = {
            'mse': [],
            'iterations': 0
        }
        
        for epoch in range(epochs):
            y_pred = X_np @ coeffs
            error = y_pred - y_np
            mse = np.mean(error**2)
            history['mse'].append(mse)
            
            if abs(prev_mse - mse) < tolerance:
                break
                
            prev_mse = mse
            gradients = (2/m) * (X_np.T @ error)
            coeffs -= lr * gradients
        
        history['iterations'] = epoch + 1
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self._training_info = {
            'method': 'gradient_descent',
            'lr': lr,
            'epochs': epochs,
            'final_epoch': epoch + 1,
            'final_mse': mse,
            'converged': epoch < epochs - 1,
            'history': history
        }

    def predict(self, X: pd.DataFrame):
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        return X_np @ np.concatenate(([self.intercept_], self.coef_))