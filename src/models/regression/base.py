import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict


class FitMethod(Enum):
    """Enum para los métodos de entrenamiento disponibles."""

    PSEUDO_INVERSE = "pseudo_inverse"
    GRADIENT_DESCENT = "gradient_descent"



class Model:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.feature_names = None
        self._training_info = {}
        

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
    
    def r2_score(self, X: pd.DataFrame, y):
        """
        Calcula el coeficiente de determinación R^2 de la predicción.
        """
        y_pred = self.predict(X)
        y_mean = y.mean()
        ss_total = ((y - y_mean) ** 2).sum()
        ss_res = ((y - y_pred) ** 2).sum()
        r2 = 1 - ss_res / ss_total
        return r2

    @staticmethod
    def _build_design_matrix(X: pd.DataFrame, degree: int = 1) -> pd.DataFrame:

        X = X.reset_index(drop=True)
        if degree == 1:
            intercept = pd.DataFrame({"intercept": np.ones(len(X))})

            return pd.concat([intercept, X.reset_index(drop=True)], axis=1)
        else:
            X_poly = pd.DataFrame({"intercept": np.ones(len(X))})
            for i in range(1, degree + 1):
                for col in X.columns:
                    X_poly[f"{col}^{i}"] = X[col] ** i
            return X_poly

    def get_coef_array(self):
        """
        Devuelve los coeficientes como un array de numpy, para mantener compatibilidad
        con código que espera ese formato.
        """
        return self._coef

    def get_coef_dict(self):
        """
        Devuelve un diccionario con los nombres de features y sus coeficientes
        """
        if hasattr(self, 'feature_names') and self.feature_names is not None and self._coef is not None:
            return dict(zip(self.feature_names, self._coef))
        return None

    def print_coefficients(self, format_precision: int = 4, metric: str = "MSE"):
        """
        Imprime los coeficientes del modelo con los nombres de sus variables.

        Parameters:
        -----------
        format_precision : int
            Número de decimales a mostrar
        metric : str
            Tipo de error a mostrar (default: "MSE")
        """
        if not hasattr(self, 'coef_dict') or self.coef_dict is None:
            return super().print_coefficients(format_precision, metric)
            
        print(f"Método: {self._training_info.get('method', 'desconocido')}")
        print(f"Intercept: {self.intercept_:.{format_precision}f}\n")
        print("Coeficientes:")
        print("-" * 30)

        for name, coef in self.coef_dict.items():
            print(f"{name:<15} | {coef:+.{format_precision}f}")

        if metric == "MSE" and "final_mse" in self._training_info:
            print(f"\nMSE final: {self._training_info['final_mse']:.{format_precision}f}")

        if metric == "R2" and "final_r2" in self._training_info:
            print(f"\nR^2 final: {self._training_info['final_r2']:.{format_precision}f}")

        if self._training_info.get("method") == "gradient_descent":
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
    
    @staticmethod
    # TODO hacer clases de loss
    def loss_mse(y_true, y_pred):
        
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        error = y_pred - y_true
        
        mse = np.mean(error**2)
        
        return mse
    
    @staticmethod
    def loss_mse_gradient(X, y_true, coeffs):
       
        m = X.shape[0]  # número de muestras
        y_pred = X @ coeffs
        error = y_pred - y_true
        
        # gradiente del MSE es (2/m) * X^T * (X*w - y)
        gradient = (2/m) * (X.T @ error)
        
        return gradient
        
    