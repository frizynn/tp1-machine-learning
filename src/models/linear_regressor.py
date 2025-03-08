from enum import Enum
from typing import Union, Optional, Dict
from models.model import Model, FitMethod
import pandas as pd
import numpy as np



class LinearRegressor(Model):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self.feature_names = None
        self._training_info = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: Union[str, FitMethod] = "pseudo_inverse",
        **kwargs
    ) -> "LinearRegressor":
        """
        Entrena el modelo usando el método especificado
        
        Parameters
        ----------
        X : pd.DataFrame
            Características de entrada
        y : pd.Series
            Variable objetivo
        method : str o FitMethod
            Método de entrenamiento ('pseudo_inverse' o 'gradient_descent')
        **kwargs : dict
            Parámetros adicionales para el método de entrenamiento seleccionado
            
        Returns
        -------
        self : LinearRegressor
            Instancia del modelo entrenado
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
            X_design = self._build_design_matrix(X, degree=1)
            self._fit_gradient_descent(X_design.values, y.values, **kwargs)

        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series):
        X_design = self._build_design_matrix(X, degree=1)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
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
        # Configuración de parámetros con valores por defecto
        params = {
            'lr': 0.01,
            'epochs': 1000,
            'tolerance': 1e-6,
            'verbose': False
        }
        
        # Actualizar con los parámetros proporcionados
        params.update(kwargs)
        
        # Extraer parámetros individuales
        lr = params['lr']
        epochs = params['epochs']
        tolerance = params['tolerance']
        verbose = params['verbose']
        
        # Validación de parámetros
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
            # Predicción y cálculo del error
            y_pred = X_np @ coeffs
            error = y_pred - y_np
            mse = np.mean(error**2)
            history["mse"].append(mse)
            
            # Mostrar progreso si verbose=True
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"Época {epoch}/{epochs}, MSE: {mse:.6f}")
            
            # Verificar convergencia
            if abs(prev_mse - mse) < tolerance:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch}")
                break
            
            prev_mse = mse
            
            # Calcular gradientes y actualizar coeficientes
            gradients = Model.loss_mse_gradient(X_np, y_np, coeffs)
            coeffs -= lr * gradients
        
        history["iterations"] = epoch + 1
        self.intercept_ = coeffs[0]
        self.coef_ = coeffs[1:]
        
        # Guardar información del entrenamiento
        self._training_info = {
            "method": "gradient_descent",
            "params": params,
            "final_epoch": epoch + 1,
            "final_mse": mse,
            "converged": epoch < epochs - 1,
            "history": history,
        }

    def predict(self, X: pd.DataFrame):
        X_design = self._build_design_matrix(X, degree=1)
        X_np = X_design.values.astype(float)
        return X_np @ np.concatenate(([self.intercept_], self.coef_))

