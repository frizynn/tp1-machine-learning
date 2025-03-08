import numpy as np
import pandas as pd
from enum import Enum
from typing import Union, Optional, Dict


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
            intercept = pd.DataFrame({"intercept": np.ones(len(X))})

            return pd.concat([intercept, X.reset_index(drop=True)], axis=1)
        else:
            X_poly = pd.DataFrame({"intercept": np.ones(len(X))})
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

        if "final_mse" in self._training_info:
            print(
                f"\nMSE final: {self._training_info['final_mse']:.{format_precision}f}"
            )

        if self._training_info.get("method") == "gradient_descent":
            print(f"Convergencia: {'Sí' if self._training_info['converged'] else 'No'}")
            print(
                f"Iteraciones: {self._training_info['final_epoch']}/{self._training_info['epochs']}"
            )

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
        
        # El gradiente del MSE es (2/m) * X^T * (X*w - y)
        gradient = (2/m) * (X.T @ error)
        
        return gradient
    


class FitMethod(Enum):
    """Enum para los métodos de entrenamiento disponibles."""

    PSEUDO_INVERSE = "pseudo_inverse"
    GRADIENT_DESCENT = "gradient_descent"

