from enum import Enum
from typing import Union, Optional, Dict
from .base import Model, FitMethod
from .polynomial_regressor import PolynomialRegressor
import pandas as pd
import numpy as np


class LinearRegressor(PolynomialRegressor):
    def __init__(self):
        super().__init__(degree=1)
    
    @classmethod
    def change_degree(cls, degree):
        raise Exception("No se puede cambiar el grado de un LinearRegressor")

