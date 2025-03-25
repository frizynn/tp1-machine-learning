from abc import ABC
from typing import Callable, Dict, Optional, Union

import numpy as np


class LossFunction(ABC):

    def __init__(self):
        """
        Initialize the loss function with MSE as the default loss.
        """
        self.default_loss = self.mse

    @classmethod
    def change_loss(cls, loss):
        """
        Change the default loss function.

        Parameters
        ----------
        loss : function
            The loss function to set as default.

        Returns
        -------
        cls
            The LossFunction class with updated default loss.
        """
        cls.default_loss = loss
        return cls

    @staticmethod
    def mse(y_true, y_pred):
        """
        Calculate the mean squared error (MSE) between y_true and y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        mse = np.mean(error**2)
        return mse

    @staticmethod
    def mae(y_true, y_pred):
        """
        Calculate the mean absolute error (MAE) between y_true and y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        mae = np.mean(np.abs(error))
        return mae

    @staticmethod
    def l2(y_true, y_pred):
        """
        Calculate the L2 norm between y_true and y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        l2 = np.linalg.norm(error)
        return l2

    @staticmethod
    def l1(y_true, y_pred):
        """
        Calculate the L1 norm between y_true and y_pred.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        error = y_pred - y_true
        l1 = np.sum(np.abs(error))
        return l1

    # Methods for regularization
    @staticmethod
    def ridge_penalty(coeffs, alpha=1.0, exclude_intercept=True):
        """
        Calculate the L2 (Ridge) penalty for coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Model coefficients
        alpha : float
            Regularization parameter
        exclude_intercept : bool
            If True, doesn't penalize the intercept (first coefficient)

        Returns
        -------
        float
            Value of the L2 penalty
        """
        if exclude_intercept and len(coeffs) > 1:
            penalty = np.sum(coeffs[1:] ** 2)
        else:
            penalty = np.sum(coeffs**2)

        return 0.5 * alpha * penalty

    @staticmethod
    def lasso_penalty(coeffs, alpha=1.0, exclude_intercept=True):
        """
        Calculate the L1 (Lasso) penalty for coefficients.

        Parameters
        ----------
        coeffs : np.ndarray
            Model coefficients
        alpha : float
            Regularization parameter
        exclude_intercept : bool
            If True, doesn't penalize the intercept (first coefficient)

        Returns
        -------
        float
            Value of the L1 penalty
        """
        if exclude_intercept and len(coeffs) > 1:
            penalty = np.sum(np.abs(coeffs[1:]))
        else:
            penalty = np.sum(np.abs(coeffs))

        return alpha * penalty

    @staticmethod
    def regularized_loss(
        loss_name,
        y_true,
        y_pred,
        coeffs,
        reg_type=None,
        alpha=0.0,
        l1_ratio=0.5,
        exclude_intercept=True,
    ):
        """
        Calculate the loss function with regularization.

        Parameters
        ----------
        loss_name : str
            Name of the loss function ('mse', 'mae', 'l1', 'l2')
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predictions
        coeffs : np.ndarray
            Model coefficients
        reg_type : str, optional
            Type of regularization ('l1', 'l2', 'elasticnet', None)
        alpha : float, optional
            Regularization parameter
        l1_ratio : float, optional
            Proportion of L1 penalty for ElasticNet
        exclude_intercept : bool, optional
            If True, doesn't penalize the intercept

        Returns
        -------
        float
            Value of the regularized loss function
        """
        # get the base loss function
        base_loss_func = getattr(LossFunction, loss_name, LossFunction.mse)
        base_loss = base_loss_func(y_true, y_pred)

        # if no regularization, return the base loss
        if reg_type is None or alpha == 0:
            return base_loss

        # calculate the penalty based on the regularization type
        if reg_type == 'l2':
            penalty = LossFunction.ridge_penalty(coeffs, alpha, exclude_intercept)
        elif reg_type == 'l1':
            penalty = LossFunction.lasso_penalty(coeffs, alpha, exclude_intercept)
        elif reg_type == 'elasticnet':
            penalty = LossFunction.elastic_net_penalty(
                coeffs, alpha, l1_ratio, exclude_intercept
            )
        else:
            raise ValueError(f"Unrecognized regularization type: {reg_type}")

        return base_loss + penalty

    @staticmethod
    def gradient(
        loss_name,
        X,
        y_true,
        coeffs,
        reg_type=None,
        alpha=0.0,
        l1_ratio=0.5,
        exclude_intercept=True,
    ):
        """
        Calculate the gradient of the specified loss function with respect to the coefficients,
        including the regularization term if specified.

        Parameters
        ----------
        loss_name : str
            Name of the loss function ('mse', 'mae', 'l1', 'l2')
        X : np.ndarray
            Design matrix with first column of ones for the intercept
        y_true : np.ndarray
            Target vector
        coeffs : np.ndarray
            Current coefficient vector
        reg_type : str, optional
            Type of regularization ('l1', 'l2', 'elasticnet', None)
        alpha : float, optional
            Regularization parameter
        l1_ratio : float, optional
            Proportion of L1 penalty for ElasticNet
        exclude_intercept : bool, optional
            If True, doesn't penalize the intercept

        Returns
        -------
        np.ndarray
            Gradient of the regularized loss function
        """
        m = X.shape[0]  # number of samples
        y_pred = X @ coeffs
        error = y_pred - y_true

        # calculate the base gradient according to the loss function
        if loss_name == 'mse':
            # gradient of MSE: (2/m) * X^T * (X*coeffs - y)
            gradient = (2 / m) * (X.T @ error)
        elif loss_name == 'mae':
            # gradient of MAE: (1/m) * X^T * sign(X*coeffs - y)
            gradient = (1 / m) * (X.T @ np.sign(error))
        elif loss_name == 'l2':
            # gradient of L2: X^T * (X*coeffs - y) / ||X*coeffs - y||_2
            norm = np.linalg.norm(error)
            if norm < 1e-10:  # Avoid division by zero
                return np.zeros_like(coeffs)
            gradient = X.T @ (error / norm)
        elif loss_name == 'l1':
            # gradient of L1: X^T * sign(X*coeffs - y)
            gradient = X.T @ np.sign(error)
        else:
            raise ValueError(f"Unrecognized loss function: {loss_name}")

        # if no regularization, return the base gradient
        if reg_type is None or alpha == 0:
            return gradient

        # calculate the gradient of the regularization term
        reg_gradient = np.zeros_like(coeffs)

        # determine which coefficients to penalize
        start_idx = 1 if exclude_intercept else 0

        if reg_type == 'l2':
            # gradient of Ridge regularization: alpha * coeffs
            reg_gradient[start_idx:] = alpha * coeffs[start_idx:]
        elif reg_type == 'l1':
            # gradient of Lasso regularization: alpha * sign(coeffs)
            reg_gradient[start_idx:] = alpha * np.sign(coeffs[start_idx:])
        elif reg_type == 'elasticnet':
            # gradient of ElasticNet regularization: combines L1 and L2
            l1_part = alpha * l1_ratio * np.sign(coeffs[start_idx:])
            l2_part = alpha * (1 - l1_ratio) * coeffs[start_idx:]
            reg_gradient[start_idx:] = l1_part + l2_part
        else:
            raise ValueError(f"Unrecognized regularization type: {reg_type}")

        return gradient + reg_gradient
