
from typing import Union, Optional
from .base import Model, FitMethod
import pandas as pd
import numpy as np
from ..loss.base import LossFunction


class PolynomialRegressor(Model):
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
        exclude_intercept: bool = True,
        report_metrics: bool = False
    ) -> "PolynomialRegressor":
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

        # validar la función de pérdida
        loss = loss.lower()
        if loss not in ['mse', 'mae', 'l1', 'l2']:
            raise ValueError(f"Función de pérdida no reconocida: {loss}")

        # validar el tipo de regularización
        if regularization is not None:
            regularization = regularization.lower()
            if regularization not in ['l1', 'l2', 'elasticnet']:
                raise ValueError(f"Tipo de regularización no reconocido: {regularization}")
            
            # validar el valor de alpha
            if alpha < 0:
                raise ValueError("El parámetro de regularización alpha debe ser mayor o igual a cero")
                
            # validar el valor de l1_ratio si usamos elasticnet
            if regularization == 'elasticnet':
                if l1_ratio < 0 or l1_ratio > 1:
                    raise ValueError("El parámetro l1_ratio debe estar entre 0 y 1")

        # almacenar los nombres originales de las features
        self.original_feature_names = X.columns.tolist()

        # para regularización L1 (Lasso) o ElasticNet con l1_ratio > 0 y pseudo_inverse,
        # lanzar error ya que requieren optimización iterativa
        if method == FitMethod.PSEUDO_INVERSE and alpha > 0:
            if regularization == 'l1':
                raise ValueError("La regularización L1 (Lasso) solo puede resolverse mediante gradient_descent, no tienen solución analítica.")

        if method == FitMethod.PSEUDO_INVERSE:
            #  si es regularización L2 (Ridge) con pseudo_inverse, usar la solución analítica
            if regularization == 'l2' and alpha > 0:
                self._fit_ridge_analytical(X, y, alpha, exclude_intercept)
            else:
                self._fit_pseudo_inverse(X, y,report_metrics)
        elif method == FitMethod.GRADIENT_DESCENT:
            X_design, poly_feature_names = self._build_design_matrix(X, degree=self.degree)
            self.feature_names = poly_feature_names[1:] 
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
                exclude_intercept=exclude_intercept,
                report_metrics=report_metrics
            )

        return self

    def _fit_ridge_analytical(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        alpha: float = 1.0, 
        exclude_intercept: bool = True    ):
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
        
        # crear matriz de identidad para regularización
        if exclude_intercept:
            identity = np.eye(n_features)
            identity[0, 0] = 0  # No regularizar el intercepto
        else:
            identity = np.eye(n_features)
        
        # solución analítica Ridge: β = (X^T·X + α·I)^(-1)·X^T·y
        XTX = X_np.T @ X_np
        XTX_reg = XTX + alpha * identity
        XTy = X_np.T @ y_np
        
        try:
            coeffs = np.linalg.solve(XTX_reg, XTy)
        except np.linalg.LinAlgError:
            coeffs = np.linalg.pinv(XTX_reg) @ XTy
        
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        self.feature_names = poly_feature_names[1:]
        
        # crear un diccionario que asigna nombres de features a coeficientes
        self.coef_dict = dict(zip(self.feature_names, self._coef))
        
        # inicializar información de entrenamiento
        training_info = {
            "method": "ridge_analytical",
            "regularization_type": "l2",
            "alpha": alpha,
            "exclude_intercept": exclude_intercept,
        }

        self._training_info = training_info
        return self

    def _fit_pseudo_inverse(self, X: pd.DataFrame, y: pd.Series, report_metrics: bool = True):
        """
        Entrena un modelo de regresión usando la pseudo-inversa.
        
        Parameters
        ----------
        X : pd.DataFrame
            Matriz de características
        y : pd.Series
            Vector objetivo
        report_metrics : bool
            Si es True, calcula y guarda métricas adicionales
        """
        X_design, poly_feature_names = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        self.feature_names = poly_feature_names[1:]  
        
        # crear un diccionario que asigna nombres de features a coeficientes
        self.coef_dict = dict(zip(self.feature_names, self._coef))
        
        self._training_info = {
            "method": "pseudo_inverse"
        }
        return self
    
    def _fit_gradient_descent(
        self,
        X_np: np.ndarray,
        y_np: np.ndarray,
        lr: float = 0.01,
        epochs: int = 1000,
        tolerance: float = 1e-6,
        verbose: bool = False,
        loss: str = 'mse',
        regularization: Optional[str] = None,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        exclude_intercept: bool = True,
        report_metrics: bool = True
    ) -> None:
        """
        Entrena el modelo mediante descenso de gradiente.
        
        Parameters
        ----------
        X_np : np.ndarray
            Matriz de características con columna de unos para el intercepto
        y_np : np.ndarray
            Vector objetivo
        lr : float
            Tasa de aprendizaje (default=0.01)
        epochs : int
            Número máximo de iteraciones (default=1000)
        tolerance : float
            Criterio de convergencia (default=1e-6)
        verbose : bool
            Mostrar progreso (default=False)
        loss : str
            Función de pérdida a utilizar (default='mse')
        regularization : str
            Tipo de regularización (default=None)
        alpha : float
            Parámetro de regularización (default=0.0)
        l1_ratio : float
            Proporción L1 para ElasticNet (default=0.5)
        exclude_intercept : bool
            No penalizar intercepto (default=True)
        """
        if loss not in ['mse', 'mae', 'l1', 'l2']:
            raise ValueError(f"Función de pérdida no reconocida: {loss}")
        
        if lr <= 0:
            raise ValueError("La tasa de aprendizaje debe ser mayor que 0")
        if epochs <= 0:
            raise ValueError("El número de épocas debe ser mayor que 0")
        if tolerance <= 0:
            raise ValueError("La tolerancia debe ser mayor que 0")
        
        # obtener la función de pérdida correspondiente
        loss_func = getattr(LossFunction, loss, LossFunction.mse)
        
        m, n = X_np.shape
        coeffs = np.zeros(n)
        prev_loss = float("inf")
        best_coeffs = coeffs.copy()
        min_loss = float("inf")
        
        history = {
            "loss": [], 
            "iterations": 0, 
            "regularization": regularization, 
            "alpha": alpha,
            "metrics": {} if report_metrics else None
        }
        
        for epoch in range(epochs):
            y_pred = X_np @ coeffs
            
            # calcular la pérdida regularizada
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
            
            gradients = LossFunction.gradient(
                loss, X_np, y_np, coeffs, 
                regularization, alpha, l1_ratio, exclude_intercept
            )
            coeffs -= lr * gradients 
            
            # rastrear los mejores coeficientes basados en la pérdida de validación
            if current_loss < min_loss:
                min_loss = current_loss
                best_coeffs = coeffs.copy()
                
       
        coeffs = best_coeffs
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        
        if self.feature_names is not None and len(self.feature_names) == len(self._coef):
            self.coef_dict = dict(zip(self.feature_names, self._coef))
        
    
            final_metrics = {}

        self._training_info = {
            "method": "gradient_descent",
            "learning_rate": lr,
            "epochs": epochs,
            "tolerance": tolerance,
            "loss_type": loss,
            "final_epoch": epoch + 1,
            "converged": abs(prev_loss - current_loss) < tolerance,
            "history": history,
            "metrics": final_metrics,
            "regularization_info": {
                "type": regularization,
                "alpha": alpha
            } if regularization else None
        }

    def _build_design_matrix(self, X: pd.DataFrame, degree: int = 1) -> tuple:
        """
        Construye la matriz de diseño para regresión polinómica.
        
        Returns
        -------
        tuple
            (DataFrame con la matriz de diseño, lista con nombres de las features polinómicas)
        """
        X = X.copy()  
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = X.reset_index(drop=True)
        
        # preparar un diccionario para recoger todas las columnas
        poly_data = {"intercept": np.ones(len(X))}
        poly_feature_names = ["intercept"]
        
        # generar términos polinómicos para cada feature
        for i in range(1, degree + 1):
            for col in X.columns:
                feature_name = f"{col}^{i}" if i > 1 else col
                poly_data[feature_name] = X[col] ** i
                poly_feature_names.append(feature_name)
        
        # crear el DataFrame de una vez
        X_poly = pd.DataFrame(poly_data)
                
        return X_poly, poly_feature_names

    @property
    def coef_(self):
        """
        Devuelve los coeficientes del modelo como un DataFrame.
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
        
        if X_np.shape[1] != len(coeffs):
            raise ValueError(f"Número incorrecto de características. Esperado: {len(coeffs)}, Recibido: {X_np.shape[1]}")
            
        return X_np @ coeffs

    def get_weights(self):
        """
        Devuelve los coeficientes del modelo como un array numpy.
        Útil para análisis y visualizaciones de pesos vs regularización.
        
        Returns
        -------
        np.ndarray
            Array con los coeficientes del modelo (sin intercepto)
        """
        if self._coef is None:
            raise ValueError("El modelo debe ser entrenado antes de obtener los pesos")
        return np.array(self._coef)

