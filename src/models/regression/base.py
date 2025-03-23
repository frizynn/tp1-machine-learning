import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict


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

    @staticmethod
    def _build_design_matrix(X: pd.DataFrame, degree: int = 1) -> pd.DataFrame:
        """
        Construye la matriz de diseño para el modelo de regresión.

        Parámetros:
        -----------
        X : pd.DataFrame
            DataFrame de características
        degree : int, default=1
            Grado de la matriz de diseño

        retorna:
        --------
        pd.DataFrame
            Matriz de diseño
        """
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
        Obtiene los coeficientes del modelo como un array de numpy.

        retorna:
        --------
        np.ndarray
            Coeficientes del modelo
        """


        return self._coef

    def get_coef_dict(self):
        """
        Obtiene un diccionario con los nombres de features y sus coeficientes.

        retorna:
        --------
        dict
            Diccionario con los nombres de features y sus coeficientes
        """
        if hasattr(self, 'feature_names') and self.feature_names is not None and self._coef is not None:
            return dict(zip(self.feature_names, self._coef))
        return None

    def print_coefficients(self, format_precision: int = 4):
        """
        Imprime los coeficientes del modelo con los nombres de sus variables.

        Parámetros:
        -----------
        format_precision : int
            Número de decimales a mostrar
        """
        if not hasattr(self, 'coef_dict') or self.coef_dict is None:
            return super().print_coefficients(format_precision)
            
        print(f"Método: {self._training_info.get('method', 'desconocido')}")
        print(f"Intercept: {self.intercept_:.{format_precision}f}\n")
        print("Coeficientes:")
        print("-" * 30)

        for name, coef in self.coef_dict.items():
            print(f"{name:<15} | {coef:+.{format_precision}f}")

        if self._training_info.get("method") == "gradient_descent":
            print(f"Convergencia: {'Sí' if self._training_info['converged'] else 'No'}")
            print(f"Iteraciones: {self._training_info['final_epoch']}/{self._training_info['epochs']}")


    def get_training_info(self) -> Dict:
        """
        Obtiene información sobre el entrenamiento del modelo.

        retorna:
        --------
        Dict
            Diccionario con información sobre el entrenamiento
        """
        return self._training_info.copy()
    
  