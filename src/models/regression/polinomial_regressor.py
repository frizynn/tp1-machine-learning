from enum import Enum
from typing import Union, Optional, Dict
from .base import Model, FitMethod
import pandas as pd
import numpy as np
from ..loss.base import LossFunction


class PolinomialRegressor(Model):
    @classmethod
    def change_degree(cls, degree):
        cls.default_degree = degree
        return cls

    def __init__(self, degree=None):
        if degree is None:
            degree = getattr(self.__class__, "default_degree", 2)
        self.degree = degree
        self._coef = None
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
        loss: str = "mse",
        regularization: Optional[str] = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        exclude_intercept: bool = True
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
        loss : str, opcional
            Función de pérdida a utilizar ('mse', 'mae', 'l1', 'l2') (default: 'mse')
        regularization : str, opcional
            Tipo de regularización ('l1', 'l2', 'elasticnet', None) (default: None)
        alpha : float, opcional
            Parámetro de regularización (default: 0.0)
        l1_ratio : float, opcional
            Proporción de la penalización L1 para ElasticNet (default: 0.5)
        exclude_intercept : bool, opcional
            Si es True, no penaliza el intercepto (default: True)
        """
        if isinstance(method, str):
            method = method.lower()
            if method == "pseudo_inverse":
                method = FitMethod.PSEUDO_INVERSE
            elif method == "gradient_descent":
                method = FitMethod.GRADIENT_DESCENT
            else:
                raise ValueError(f"Método no reconocido: {method}")

        # Validar la función de pérdida
        loss = loss.lower()
        if loss not in ['mse', 'mae', 'l1', 'l2']:
            raise ValueError(f"Función de pérdida no reconocida: {loss}")

        # Validar el tipo de regularización
        if regularization is not None:
            regularization = regularization.lower()
            if regularization not in ['l1', 'l2', 'elasticnet']:
                raise ValueError(f"Tipo de regularización no reconocido: {regularization}")
            
            # Validar el valor de alpha
            if alpha < 0:
                raise ValueError("El parámetro de regularización alpha debe ser mayor o igual a cero")
                
            # Validar el valor de l1_ratio si usamos elasticnet
            if regularization == 'elasticnet':
                if l1_ratio < 0 or l1_ratio > 1:
                    raise ValueError("El parámetro l1_ratio debe estar entre 0 y 1")

        # Store original feature names
        self.original_feature_names = X.columns.tolist()

        # Para regularización L1 (Lasso) o ElasticNet con l1_ratio > 0 y pseudo_inverse,
        # lanzar error ya que requieren optimización iterativa
        if method == FitMethod.PSEUDO_INVERSE and alpha > 0:
            if regularization == 'l1' or (regularization == 'elasticnet' and l1_ratio > 0):
                raise ValueError("La regularización L1 (Lasso) y ElasticNet solo pueden resolverse "
                                "mediante gradient_descent, no tienen solución analítica.")

        if method == FitMethod.PSEUDO_INVERSE:
            # Si es regularización L2 (Ridge) con pseudo_inverse, usar la solución analítica
            if regularization == 'l2' and alpha > 0:
                self._fit_ridge_analytical(X, y, alpha, exclude_intercept)
            else:
                self._fit_pseudo_inverse(X, y)
        elif method == FitMethod.GRADIENT_DESCENT:
            X_design, poly_feature_names = self._build_design_matrix(X, degree=self.degree)
            self.feature_names = poly_feature_names[1:]  # Exclude intercept
            self._fit_gradient_descent(
                X_design.values,
                y.values,
                lr=learning_rate,
                epochs=epochs,
                tolerance=tolerance,
                verbose=verbose,
                loss=loss,
                regularization=regularization,
                alpha=alpha,
                l1_ratio=l1_ratio,
                exclude_intercept=exclude_intercept
            )

        return self

    def _fit_ridge_analytical(self, X: pd.DataFrame, y: pd.Series, alpha: float = 1.0, exclude_intercept: bool = True):
        """
        Entrena un modelo de regresión Ridge usando la solución analítica.
        
        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características
        y : pd.Series
            Vector objetivo
        alpha : float
            Parámetro de regularización
        exclude_intercept : bool
            Si es True, no penaliza el intercepto
        """
        X_design, poly_feature_names = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)
        n_samples, n_features = X_np.shape
        
        # Crear matriz de identidad para regularización
        # Si exclude_intercept es True, no penalizamos el intercepto (primera columna)
        if exclude_intercept:
            identity = np.eye(n_features)
            identity[0, 0] = 0  # No regularizar el intercepto
        else:
            identity = np.eye(n_features)
        
        # Solución analítica Ridge: β = (X^T·X + α·I)^(-1)·X^T·y
        XTX = X_np.T @ X_np
        XTX_reg = XTX + alpha * identity
        XTy = X_np.T @ y_np
        
        try:
            # Intentar resolver con inversión directa
            coeffs = np.linalg.solve(XTX_reg, XTy)
        except np.linalg.LinAlgError:
            # Si hay problemas numéricos, usar pseudoinversa
            coeffs = np.linalg.pinv(XTX_reg) @ XTy
        
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        self.feature_names = poly_feature_names[1:]  # Exclude intercept
        
        # Create a dictionary mapping feature names to coefficients
        self.coef_dict = dict(zip(self.feature_names, self._coef))
        
        # Calcular métricas finales
        y_pred = X_np @ coeffs
        final_mse = LossFunction.mse(y_np, y_pred)
        final_r2 = 1 - ((y_np - y_pred) ** 2).sum() / ((y_np - y_np.mean()) ** 2).sum()
        
        # Calcular la pérdida regularizada para el seguimiento
        reg_loss = final_mse
        if alpha > 0:
            ridge_penalty = LossFunction.ridge_penalty(coeffs, alpha, exclude_intercept)
            reg_loss = final_mse + ridge_penalty
        
        self._training_info = {
            "method": "ridge_analytical",
            "regularization_type": "l2",
            "alpha": alpha,
            "exclude_intercept": exclude_intercept,
            "final_mse": final_mse,
            "final_r2": final_r2,
            "final_reg_mse": reg_loss
        }
        
        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series):
        X_design, poly_feature_names = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, residuals, rank, s = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        self.feature_names = poly_feature_names[1:]  # Exclude intercept
        
        # Create a dictionary mapping feature names to coefficients
        self.coef_dict = dict(zip(self.feature_names, self._coef))
        
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
                - loss (str): Función de pérdida a utilizar (default='mse')
                - regularization (str): Tipo de regularización (default=None)
                - alpha (float): Parámetro de regularización (default=0.0)
                - l1_ratio (float): Proporción L1 para ElasticNet (default=0.5)
                - exclude_intercept (bool): No penalizar intercepto (default=True)
        """
        params = {
            'lr': 0.01,
            'epochs': 1000,
            'tolerance': 1e-6,
            'verbose': False,
            'loss': 'mse',
            'regularization': None,
            'alpha': 0.0,
            'l1_ratio': 0.5,
            'exclude_intercept': True
        }
        
        params.update(kwargs)
        
        lr = params['lr']
        epochs = params['epochs']
        tolerance = params['tolerance']
        verbose = params['verbose']
        loss = params['loss']
        regularization = params['regularization']
        alpha = params['alpha']
        l1_ratio = params['l1_ratio']
        exclude_intercept = params['exclude_intercept']
        
        if lr <= 0:
            raise ValueError("La tasa de aprendizaje debe ser mayor que 0")
        if epochs <= 0:
            raise ValueError("El número de épocas debe ser mayor que 0")
        if tolerance <= 0:
            raise ValueError("La tolerancia debe ser mayor que 0")
        
        # Obtener la función de pérdida correspondiente
        loss_func = getattr(LossFunction, loss, LossFunction.mse)
        
        m, n = X_np.shape
        coeffs = np.zeros(n)
        prev_loss = float("inf")
        
        history = {
            "loss": [], 
            "iterations": 0, 
            "regularization": regularization, 
            "alpha": alpha
        }
        
        for epoch in range(epochs):
            y_pred = X_np @ coeffs
            
            # Calcular la pérdida regularizada
            if regularization is not None and alpha > 0:
                current_loss = LossFunction.regularized_loss(
                    loss, y_np, y_pred, coeffs, 
                    regularization, alpha, l1_ratio, exclude_intercept
                )
            else:
                current_loss = loss_func(y_np, y_pred)
                
            history["loss"].append(current_loss)
            
            if verbose and (epoch % max(1, epochs // 10) == 0):
                loss_name = loss.upper()
                reg_info = f" + {regularization.upper()}(α={alpha:.4f})" if regularization else ""
                print(f"Época {epoch}/{epochs}, {loss_name}{reg_info}: {current_loss:.6f}")
            
            if abs(prev_loss - current_loss) < tolerance:
                if verbose:
                    print(f"Convergencia alcanzada en época {epoch}")
                break
            
            prev_loss = current_loss
            
            # Calcular gradientes con regularización
            gradients = LossFunction.gradient(
                loss, X_np, y_np, coeffs, 
                regularization, alpha, l1_ratio, exclude_intercept
            )
            coeffs -= lr * gradients
        
        history["iterations"] = epoch + 1
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        
        if self.feature_names is not None and len(self.feature_names) == len(self._coef):
            self.coef_dict = dict(zip(self.feature_names, self._coef))
        
        # Calc final metrics
        y_pred = X_np @ coeffs
        
        # Calcular la pérdida final sin regularización para reportar métricas puras
        final_loss = loss_func(y_np, y_pred)
        final_mse = LossFunction.mse(y_np, y_pred)
        
        # Si tenemos regularización, también calculamos la pérdida regularizada final
        reg_info = {}
        if regularization is not None and alpha > 0:
            reg_loss = LossFunction.regularized_loss(
                loss, y_np, y_pred, coeffs, 
                regularization, alpha, l1_ratio, exclude_intercept
            )
            reg_info = {
                "regularization_type": regularization,
                "alpha": alpha,
                "l1_ratio": l1_ratio if regularization == 'elasticnet' else None,
                f"final_reg_{loss}": reg_loss
            }

        self._training_info = {
            "method": "gradient_descent",
            "params": params,
            "final_epoch": epoch + 1,
            "epochs": epochs,
            "converged": abs(prev_loss - current_loss) < tolerance,
            "history": history,
            "loss_type": loss,
            f"final_{loss}": final_loss,
            "final_mse": final_mse,
            **reg_info
        }   
    
    def _build_design_matrix(self, X: pd.DataFrame, degree: int = 1) -> tuple:
        """
        Construye la matriz de diseño para regresión polinómica.
        
        Returns
        -------
        tuple
            (DataFrame con la matriz de diseño, lista con nombres de las features polinómicas)
        """
        X = X.copy()  # Para evitar modificar el DataFrame original
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        # Asegurarse de que X sea DataFrame
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = X.reset_index(drop=True)
        
        # Prepare a dictionary to collect all columns
        poly_data = {"intercept": np.ones(len(X))}
        poly_feature_names = ["intercept"]
        
        # Generar términos polinómicos para cada feature
        for i in range(1, degree + 1):
            for col in X.columns:
                feature_name = f"{col}^{i}" if i > 1 else col
                poly_data[feature_name] = X[col] ** i
                poly_feature_names.append(feature_name)
        
        # Create the DataFrame all at once
        X_poly = pd.DataFrame(poly_data)
                
        return X_poly, poly_feature_names

    @property
    def coef_(self):
        """
        Devuelve los coeficientes del modelo.
        Si se ha entrenado el modelo, devuelve un DataFrame con los nombres de features y sus coeficientes.
        De lo contrario, devuelve None.
        """
        if hasattr(self, '_coef') and self._coef is not None and hasattr(self, 'feature_names'):
            return pd.DataFrame({
                'feature': self.feature_names,
                'coefficient': self._coef
            })
        return self._coef
    
    @coef_.setter
    def coef_(self, value):
        """
        Establece los coeficientes del modelo.
        """
        self._coef = value

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
        if self._coef is None or self.intercept_ is None:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
            
        X_design, _ = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        coeffs = np.concatenate(([self.intercept_], self._coef))
        
        # Asegurarse de que las dimensiones coincidan
        if X_np.shape[1] != len(coeffs):
            raise ValueError(f"Número incorrecto de características. Esperado: {len(coeffs)}, Recibido: {X_np.shape[1]}")
            
        return X_np @ coeffs

