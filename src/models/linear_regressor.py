from enum import Enum
from typing import Union, Optional, Dict
from models.model import Model, FitMethod
from models.polinomial_regressor import PolinomialRegressor
import pandas as pd
import numpy as np


class LinearRegressor(PolinomialRegressor):
    def __init__(self):
        super().__init__(degree=1)
    

