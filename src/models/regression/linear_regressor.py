from .polynomial_regressor import PolynomialRegressor


class LinearRegressor(PolynomialRegressor):
    """
    Linear regression model (polynomial of degree 1).
    Extends the PolynomialRegressor with a fixed degree of 1.
    """
    def __init__(self):
        """
        Initialize a linear regressor with degree=1
        """
        super().__init__(degree=1)
    
    @classmethod
    def change_degree(cls, degree):
        """
        Method overridden to prevent changing the degree of a LinearRegressor.
        
        Parameters:
        -----------
        degree : int
            Not used, raises an exception
            
        Raises:
        -------
        Exception
            Always raises an exception since degree cannot be changed
        """
        raise Exception("Cannot change the degree of a LinearRegressor")

