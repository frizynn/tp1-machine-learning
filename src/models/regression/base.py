from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd


class FitMethod(Enum):
    """Enum for the available training methods."""

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
        Builds the design matrix for the regression model.

        Parameters:
        -----------
        X : pd.DataFrame
            DataFrame of features
        degree : int, default=1
            Degree of the design matrix

        Returns:
        --------
        pd.DataFrame
            Design matrix
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
        Gets the model coefficients as a numpy array.

        Returns:
        --------
        np.ndarray
            Model coefficients
        """


        return self._coef

    def get_coef_dict(self):
        """
        Gets a dictionary with feature names and their coefficients.

        Returns:
        --------
        dict
            Dictionary with feature names and their coefficients
        """
        if hasattr(self, 'feature_names') and self.feature_names is not None and self._coef is not None:
            return dict(zip(self.feature_names, self._coef))
        return None

    def print_coefficients(self, format_precision: int = 4):
        """
        Prints the model coefficients with their variable names.

        Parameters:
        -----------
        format_precision : int
            Number of decimal places to display
        """
        if not hasattr(self, 'coef_dict') or self.coef_dict is None:
            return super().print_coefficients(format_precision)
            
        print(f"Method: {self._training_info.get('method', 'unknown')}")
        print(f"Intercept: {self.intercept_:.{format_precision}f}\n")
        print("Coefficients:")
        print("-" * 30)

        for name, coef in self.coef_dict.items():
            print(f"{name:<15} | {coef:+.{format_precision}f}")

        if self._training_info.get("method") == "gradient_descent":
            print(f"Convergence: {'Yes' if self._training_info['converged'] else 'No'}")
            print(f"Iterations: {self._training_info['final_epoch']}/{self._training_info['epochs']}")


    def get_training_info(self) -> Dict:
        """
        Gets information about the model training.

        Returns:
        --------
        Dict
            Dictionary with information about the training
        """
        return self._training_info.copy()
    
  