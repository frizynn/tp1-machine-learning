from enum import Enum
from typing import Union, Optional, Dict
from .base import Model, FitMethod
import pandas as pd
import numpy as np


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

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Union[str, FitMethod] = "pseudo_inverse",
        learning_rate: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
    ) -> "PolinomialRegressor":
        """
        Entrena el modelo usando el método especificado

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características
        y : pd.Series
            Vector objetivo
        method : Union[str, FitMethod]
            Método de entrenamiento ('pseudo_inverse' o 'gradient_descent')
        learning_rate : float, opcional
            Tasa de aprendizaje para descenso de gradiente (default: 0.01)
        epochs : int, opcional
            Número máximo de iteraciones para descenso de gradiente (default: 1000)
        tolerance : float, opcional
            Criterio de convergencia para descenso de gradiente (default: 1e-6)
        verbose : bool, opcional
            Mostrar progreso durante el entrenamiento (default: False)
        """
        if isinstance(method, str):
            method = method.lower()
            if method == "pseudo_inverse":
                method = FitMethod.PSEUDO_INVERSE
            elif method == "gradient_descent":
                method = FitMethod.GRADIENT_DESCENT
            else:
                raise ValueError(f"Método no reconocido: {method}")

        if method == FitMethod.PSEUDO_INVERSE:
            if any(param != default for param, default in [
                (learning_rate, 0.01),
                (epochs, 1000),
                (tolerance, 1e-6),
                (verbose, False)
            ]):
                raise ValueError(
                    "Los parámetros learning_rate, epochs, tolerance y verbose no deben "
                    "especificarse cuando se usa el método pseudo_inverse"
                )

        self.feature_names = X.columns

        if method == FitMethod.PSEUDO_INVERSE:
            self._fit_pseudo_inverse(X, y)
        elif method == FitMethod.GRADIENT_DESCENT:
            X_design = self._build_design_matrix(X, degree=self.degree)
            self._fit_gradient_descent(
                X_design.values,
                y.values,
                lr=learning_rate,
                epochs=epochs,
                tolerance=tolerance,
                verbose=verbose
            )

        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        self._training_info = {
            "method": "pseudo_inverse",
            "final_mse": self.mse_score(X, y),
        }
        return self

    def _fit_gradient_descent(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        **kwargs
    ) -> None:
        """
        Entrena el modelo mediante descenso de gradiente.
        
        Parameters
        ----------
        X_np : np.ndarray
            Matriz de características con columna de unos para el intercepto
        y_np : np.ndarray
            Vector objetivo
        **kwargs : dict
            Parámetros opcionales del descenso de gradiente:
                - lr (float): Tasa de aprendizaje (default=0.01)
                - epochs (int): Número máximo de iteraciones (default=1000)
                - tolerance (float): Criterio de convergencia (default=1e-6)
                - verbose (bool): Mostrar progreso (default=False)
        """
        params = {
            'lr': 0.01,
            'epochs': 1000,
            'tolerance': 1e-6,
            'verbose': False
        }
        
        params.update(kwargs)
        
        lr = params['lr']
        epochs = params['epochs']
        tolerance = params['tolerance']
        verbose = params['verbose']
        
        if lr <= 0:
            raise ValueError("La tasa de aprendizaje debe ser mayor que 0")
        if epochs <= 0:
            raise ValueError("El número de épocas debe ser mayor que 0")
        if tolerance <= 0:
            raise ValueError("La tolerancia debe ser mayor que 0")
        
        m, n = X_np.shape
        coeffs = np.zeros(n)
        prev_mse = float("inf")
        
        history = {"mse": [], "iterations": 0}
        
        for epoch in range(epochs):
            y_pred = X_np @ coeffs
            error = y_pred - y_np
            mse = np.mean(error**2)
            history["mse"].append(mse)
            
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Época {epoch}/{epochs}, MSE: {mse:.6f}")
            
            if abs(prev_mse - mse) < tolerance:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch}")
                break
            
            prev_mse = mse
            gradients = Model.loss_mse_gradient(X_np, y_np, coeffs)
            coeffs -= lr * gradients
        
        history["iterations"] = epoch + 1
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        
        # Convert X_np back to DataFrame for r2_score calculation
        X_df = pd.DataFrame(X_np[:, 1:])  
        self._training_info = {
            "method": "gradient_descent",
            "params": params,
            "final_epoch": epoch + 1,
            "epochs": epochs,
            "final_mse": mse,
            "final_r2": self.r2_score(X_df, pd.Series(y_np)),
            "converged": abs(prev_mse - mse) < tolerance,
            "history": history,
        }

    def _build_design_matrix(self, X: pd.DataFrame, degree: int = 1) -> pd.DataFrame:
        """
        Construye la matriz de diseño para regresión polinómica.
        """
        X = X.copy()  # Para evitar modificar el DataFrame original
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        # Asegurarse de que X sea DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = X.reset_index(drop=True)
        X_poly = pd.DataFrame({"intercept": np.ones(len(X))})
        
        # Generar términos polinómicos para cada feature
        for i in range(1, degree + 1):
            for col in X.columns:
                X_poly[f"{col}^{i}"] = X[col] ** i
                
        return X_poly

    def predict(self, X: pd.DataFrame):
        """
        Realiza predicciones sobre nuevas muestras.

        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características

        Returns
        -------
        np.ndarray
            Vector de predicciones
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
            
        X_design = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        coeffs = np.concatenate(([self.intercept_], self.coef_))
        
        # Asegurarse de que las dimensiones coincidan
        if X_np.shape[1] != len(coeffs):
            raise ValueError(f"Número incorrecto de características. Esperado: {len(coeffs)}, Recibido: {X_np.shape[1]}")
            
        return X_np @ coeffs

