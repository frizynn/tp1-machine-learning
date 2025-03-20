from abc import ABC
import numpy as np
from typing import Optional, Callable, Dict, Union


class LossFunction(ABC):

    def __init__(self):
        self.default_loss = self.mse

    @classmethod
    def change_loss(cls, loss):
        cls.default_loss = loss
        return cls

    @staticmethod
    def mse(y_true, y_pred):
        """
        Calcula el error cuadrático medio (MSE) entre y_true e y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        mse = np.mean(error ** 2)
        return mse
    
    @staticmethod
    def mae(y_true, y_pred):
        """
        Calcula el error absoluto medio (MAE) entre y_true e y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        mae = np.mean(np.abs(error))
        return mae
    
    @staticmethod
    def l2(y_true, y_pred):
        """
        Calcula la norma L2 entre y_true e y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        l2 = np.linalg.norm(error)
        return l2
    
    @staticmethod
    def l1(y_true, y_pred):
        """
        Calcula la norma L1 entre y_true e y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        l1 = np.sum(np.abs(error))
        return l1
    
    # Métodos para regularización
    @staticmethod
    def ridge_penalty(coeffs, alpha=1.0, exclude_intercept=True):
        """
        Calcula la penalización L2 (Ridge) para los coeficientes.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Coeficientes del modelo
        alpha : float
            Parámetro de regularización
        exclude_intercept : bool
            Si es True, no penaliza el intercepto (primer coeficiente)
            
        Returns
        -------
        float
            Valor de la penalización L2
        """
        if exclude_intercept and len(coeffs) > 1:
            penalty = np.sum(coeffs[1:]**2)
        else:
            penalty = np.sum(coeffs**2)
            
        return 0.5 * alpha * penalty
    
    @staticmethod
    def lasso_penalty(coeffs, alpha=1.0, exclude_intercept=True):
        """
        Calcula la penalización L1 (Lasso) para los coeficientes.
        
        Parameters
        ----------
        coeffs : np.ndarray
            Coeficientes del modelo
        alpha : float
            Parámetro de regularización
        exclude_intercept : bool
            Si es True, no penaliza el intercepto (primer coeficiente)
            
        Returns
        -------
        float
            Valor de la penalización L1
        """
        if exclude_intercept and len(coeffs) > 1:
            penalty = np.sum(np.abs(coeffs[1:]))
        else:
            penalty = np.sum(np.abs(coeffs))
            
        return alpha * penalty
    
    @staticmethod
    def elastic_net_penalty(coeffs, alpha=1.0, l1_ratio=0.5, exclude_intercept=True):
        """
        Calcula la penalización de ElasticNet (combinación de L1 y L2).
        
        Parameters
        ----------
        coeffs : np.ndarray
            Coeficientes del modelo
        alpha : float
            Parámetro de regularización global
        l1_ratio : float
            Proporción de la penalización L1 (0 = Ridge, 1 = Lasso)
        exclude_intercept : bool
            Si es True, no penaliza el intercepto (primer coeficiente)
            
        Returns
        -------
        float
            Valor de la penalización ElasticNet
        """
        lasso = LossFunction.lasso_penalty(coeffs, alpha * l1_ratio, exclude_intercept)
        ridge = LossFunction.ridge_penalty(coeffs, alpha * (1 - l1_ratio), exclude_intercept)
        return lasso + ridge
    
    @staticmethod
    def regularized_loss(loss_name, y_true, y_pred, coeffs, 
                         reg_type=None, alpha=0.0, l1_ratio=0.5, exclude_intercept=True):
        """
        Calcula la función de pérdida con regularización.
        
        Parameters
        ----------
        loss_name : str
            Nombre de la función de pérdida ('mse', 'mae', 'l1', 'l2')
        y_true : np.ndarray
            Valores reales
        y_pred : np.ndarray
            Predicciones
        coeffs : np.ndarray
            Coeficientes del modelo
        reg_type : str, optional
            Tipo de regularización ('l1', 'l2', 'elasticnet', None)
        alpha : float, optional
            Parámetro de regularización
        l1_ratio : float, optional
            Proporción de la penalización L1 para ElasticNet
        exclude_intercept : bool, optional
            Si es True, no penaliza el intercepto
            
        Returns
        -------
        float
            Valor de la función de pérdida regularizada
        """
        # Obtener la función de pérdida base
        base_loss_func = getattr(LossFunction, loss_name, LossFunction.mse)
        base_loss = base_loss_func(y_true, y_pred)
        
        # Si no hay regularización, devolver la pérdida base
        if reg_type is None or alpha == 0:
            return base_loss
        
        # Calcular la penalización según el tipo de regularización
        if reg_type == 'l2':
            penalty = LossFunction.ridge_penalty(coeffs, alpha, exclude_intercept)
        elif reg_type == 'l1':
            penalty = LossFunction.lasso_penalty(coeffs, alpha, exclude_intercept)
        elif reg_type == 'elasticnet':
            penalty = LossFunction.elastic_net_penalty(coeffs, alpha, l1_ratio, exclude_intercept)
        else:
            raise ValueError(f"Tipo de regularización no reconocido: {reg_type}")
        
        return base_loss + penalty
    
    @staticmethod
    def gradient(loss_name, X, y_true, coeffs, reg_type=None, alpha=0.0, l1_ratio=0.5, exclude_intercept=True):
        """
        Calcula el gradiente de la función de pérdida especificada respecto a los coeficientes,
        incluyendo el término de regularización si se especifica.
        
        Parameters
        ----------
        loss_name : str
            Nombre de la función de pérdida ('mse', 'mae', 'l1', 'l2')
        X : np.ndarray
            Matriz de diseño con primera columna de unos para el intercepto
        y_true : np.ndarray
            Vector objetivo
        coeffs : np.ndarray
            Vector de coeficientes actual
        reg_type : str, optional
            Tipo de regularización ('l1', 'l2', 'elasticnet', None)
        alpha : float, optional
            Parámetro de regularización
        l1_ratio : float, optional
            Proporción de la penalización L1 para ElasticNet
        exclude_intercept : bool, optional
            Si es True, no penaliza el intercepto
            
        Returns
        -------
        np.ndarray
            Gradiente de la función de pérdida regularizada
        """
        m = X.shape[0]  # número de muestras
        y_pred = X @ coeffs
        error = y_pred - y_true
        
        # Calcular el gradiente base según la función de pérdida
        if loss_name == 'mse':
            # gradiente del MSE: (2/m) * X^T * (X*coeffs - y)
            gradient = (2/m) * (X.T @ error)
        elif loss_name == 'mae':
            # gradiente del MAE: (1/m) * X^T * sign(X*coeffs - y)
            gradient = (1/m) * (X.T @ np.sign(error))
        elif loss_name == 'l2':
            # gradiente de L2: X^T * (X*coeffs - y) / ||X*coeffs - y||_2
            norm = np.linalg.norm(error)
            if norm < 1e-10:  # Evitar división por cero
                return np.zeros_like(coeffs)
            gradient = X.T @ (error / norm)
        elif loss_name == 'l1':
            # gradiente de L1: X^T * sign(X*coeffs - y)
            gradient = X.T @ np.sign(error)
        else:
            raise ValueError(f"Función de pérdida no reconocida: {loss_name}")
            
        # Si no hay regularización, devolver el gradiente base
        if reg_type is None or alpha == 0:
            return gradient
        
        # Calcular el gradiente del término de regularización
        reg_gradient = np.zeros_like(coeffs)
        
        # Determinar qué coeficientes penalizar
        start_idx = 1 if exclude_intercept else 0
        
        if reg_type == 'l2':
            # gradiente de la regularización Ridge: alpha * coeffs
            reg_gradient[start_idx:] = alpha * coeffs[start_idx:]
        elif reg_type == 'l1':
            # gradiente de la regularización Lasso: alpha * sign(coeffs)
            reg_gradient[start_idx:] = alpha * np.sign(coeffs[start_idx:])
        elif reg_type == 'elasticnet':
            # gradiente de la regularización ElasticNet: combina L1 y L2
            l1_part = alpha * l1_ratio * np.sign(coeffs[start_idx:])
            l2_part = alpha * (1 - l1_ratio) * coeffs[start_idx:]
            reg_gradient[start_idx:] = l1_part + l2_part
        else:
            raise ValueError(f"Tipo de regularización no reconocido: {reg_type}")
            
        return gradient + reg_gradient
    