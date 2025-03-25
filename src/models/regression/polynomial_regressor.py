from typing import Optional, Union

import numpy as np
import pandas as pd

from ..loss.base import LossFunction
from .base import FitMethod, Model


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
        Trains the model using the specified method

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        method : Union[str, FitMethod]
            Training method ('pseudo_inverse' or 'gradient_descent')
        learning_rate : float, optional
            Learning rate for gradient descent (default: 0.01)
        epochs : int, optional
            Maximum number of iterations for gradient descent (default: 1000)
        tolerance : float, optional
            Convergence criterion for gradient descent (default: 1e-6)
        verbose : bool, optional
            Show progress during training (default: False)
        loss : str, optional
            Loss function to use ('mse', 'mae', 'l1', 'l2') (default: 'mse')
        regularization : str, optional
            Type of regularization ('l1', 'l2', 'elasticnet', None) (default: None)
        alpha : float, optional
            Regularization parameter (default: 0.0)
        l1_ratio : float, optional
            Proportion of L1 penalty for ElasticNet (default: 0.5)
        exclude_intercept : bool, optional
            If True, does not penalize the intercept (default: True)
        """
        if isinstance(method, str):
            method = method.lower()
            if method == "pseudo_inverse":
                method = FitMethod.PSEUDO_INVERSE
            elif method == "gradient_descent":
                method = FitMethod.GRADIENT_DESCENT
            else:
                raise ValueError(f"Unrecognized method: {method}")

        # validate the loss function
        loss = loss.lower()
        if loss not in ['mse', 'mae', 'l1', 'l2']:
            raise ValueError(f"Unrecognized loss function: {loss}")

        # validate the regularization type
        if regularization is not None:
            regularization = regularization.lower()
            if regularization not in ['l1', 'l2', 'elasticnet']:
                raise ValueError(f"Unrecognized regularization type: {regularization}")
            
            # validate alpha value
            if alpha < 0:
                raise ValueError("The regularization parameter alpha must be greater than or equal to zero")
                
            # validate l1_ratio value if using elasticnet
            if regularization == 'elasticnet':
                if l1_ratio < 0 or l1_ratio > 1:
                    raise ValueError("The l1_ratio parameter must be between 0 and 1")

        # store the original feature names
        self.original_feature_names = X.columns.tolist()

        # for L1 (Lasso) regularization or ElasticNet with l1_ratio > 0 and pseudo_inverse,
        # raise error since they require iterative optimization
        if method == FitMethod.PSEUDO_INVERSE and alpha > 0:
            if regularization == 'l1':
                raise ValueError("L1 (Lasso) regularization can only be solved using gradient_descent, they don't have an analytical solution.")

        if method == FitMethod.PSEUDO_INVERSE:
            # if it's L2 (Ridge) regularization with pseudo_inverse, use the analytical solution
            if regularization == 'l2' and alpha > 0:
                self._fit_ridge_analytical(X, y, alpha, exclude_intercept)
            else:
                self._fit_pseudo_inverse(X, y, report_metrics)
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
        Trains a Ridge regression model using the analytical solution.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        alpha : float
            Regularization parameter
        exclude_intercept : bool
            If True, does not penalize the intercept
       
        """
        X_design, poly_feature_names = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)
        n_samples, n_features = X_np.shape
        
        # create identity matrix for regularization
        if exclude_intercept:
            identity = np.eye(n_features)
            identity[0, 0] = 0  # Do not regularize the intercept
        else:
            identity = np.eye(n_features)
        
        # Ridge analytical solution: β = (X^T·X + α·I)^(-1)·X^T·y
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
        
        # create a dictionary that maps feature names to coefficients
        self.coef_dict = dict(zip(self.feature_names, self._coef))
        
        # initialize training information
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
        Trains a regression model using the pseudo-inverse.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        report_metrics : bool
            If True, calculates and stores additional metrics
        """
        X_design, poly_feature_names = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        y_np = y.values.astype(float)

        coeffs, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
        self.intercept_ = coeffs[0]
        self._coef = coeffs[1:]
        self.feature_names = poly_feature_names[1:]  
        
        # create a dictionary that maps feature names to coefficients
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
        Trains the model using gradient descent.
        
        Parameters
        ----------
        X_np : np.ndarray
            Feature matrix with a column of ones for the intercept
        y_np : np.ndarray
            Target vector
        lr : float
            Learning rate (default=0.01)
        epochs : int
            Maximum number of iterations (default=1000)
        tolerance : float
            Convergence criterion (default=1e-6)
        verbose : bool
            Show progress (default=False)
        loss : str
            Loss function to use (default='mse')
        regularization : str
            Type of regularization (default=None)
        alpha : float
            Regularization parameter (default=0.0)
        l1_ratio : float
            L1 ratio for ElasticNet (default=0.5)
        exclude_intercept : bool
            Do not penalize intercept (default=True)
        """
        if loss not in ['mse', 'mae', 'l1', 'l2']:
            raise ValueError(f"Unrecognized loss function: {loss}")
        
        if lr <= 0:
            raise ValueError("The learning rate must be greater than 0")
        if epochs <= 0:
            raise ValueError("The number of epochs must be greater than 0")
        if tolerance <= 0:
            raise ValueError("The tolerance must be greater than 0")
        
        # get the corresponding loss function
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
            
            # calculate the regularized loss
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
                print(f"Epoch {epoch}/{epochs}, {loss_name}{reg_info}: {current_loss:.6f}")
            
            if abs(prev_loss - current_loss) < tolerance:
                if verbose:
                    print(f"Convergence achieved at epoch {epoch}")
                break
            
            prev_loss = current_loss
            
            gradients = LossFunction.gradient(
                loss, X_np, y_np, coeffs, 
                regularization, alpha, l1_ratio, exclude_intercept
            )
            coeffs -= lr * gradients 
            
            # track the best coefficients based on validation loss
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
        Builds the design matrix for polynomial regression.
        
        Returns
        -------
        tuple
            (DataFrame with the design matrix, list with polynomial feature names)
        """
        X = X.copy()  
        if isinstance(X, pd.Series):
            X = X.to_frame()
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X = X.reset_index(drop=True)
        
        # prepare a dictionary to collect all columns
        poly_data = {"intercept": np.ones(len(X))}
        poly_feature_names = ["intercept"]
        
        # generate polynomial terms for each feature
        for i in range(1, degree + 1):
            for col in X.columns:
                feature_name = f"{col}^{i}" if i > 1 else col
                poly_data[feature_name] = X[col] ** i
                poly_feature_names.append(feature_name)
        
        # create the DataFrame at once
        X_poly = pd.DataFrame(poly_data)
                
        return X_poly, poly_feature_names

    @property
    def coef_(self):
        """
        Returns the model coefficients as a DataFrame.
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
        Sets the model coefficients.
        """
        self._coef = value

    def predict(self, X: pd.DataFrame):
        """
        Makes predictions on new samples.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix

        Returns
        -------
        np.ndarray
            Prediction vector
        """
        if self._coef is None or self.intercept_ is None:
            raise ValueError("The model must be trained before making predictions")
            
        X_design, _ = self._build_design_matrix(X, self.degree)
        X_np = X_design.values.astype(float)
        coeffs = np.concatenate(([self.intercept_], self._coef))
        
        if X_np.shape[1] != len(coeffs):
            raise ValueError(f"Incorrect number of features. Expected: {len(coeffs)}, Received: {X_np.shape[1]}")
            
        return X_np @ coeffs

    def get_weights(self):
        """
        Returns the model coefficients as a numpy array.
        Useful for analysis and visualizations of weights vs regularization.
        
        Returns
        -------
        np.ndarray
            Array with the model coefficients (without intercept)
        """
        if self._coef is None:
            raise ValueError("The model must be trained before getting the weights")
        return np.array(self._coef)

