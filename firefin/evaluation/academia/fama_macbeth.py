import pandas as pd
from typing import List
from ...core.algorithm.regression import RollingRegressor, BatchRegressionResult

class FamaMacBeth:

    @staticmethod
    def run_regression(factor: pd.DataFrame|pd.Series, return_adj: pd.DataFrame, window: int = 252) -> BatchRegressionResult:
        """
        Run Fama-MacBeth regression."
        """
        if isinstance(factor, pd.Series):
            # Convert series to DataFrame for consistency
            factor = pd.concat([factor] * return_adj.shape[1], axis=1)
            factor.columns = return_adj.columns
        elif isinstance(factor, pd.DataFrame):
            pass
        else:
            raise ValueError("Factor must be a pandas Series or DataFrame.")
        
        # Note: Calculate excess returns if necessary
        # return_adj = return_adj - risk_free_rate
        # excess return is different in many cases, we leave it to the user to handle this.

        # First step: Time-series regressions
        r = RollingRegressor(factor, return_adj, None, fit_intercept=True).fit(window)

        # Second step: Cross-sectional regressions
        # This step involves regressing the time-series regression coefficients on the factors
        r = RollingRegressor(r.beta, return_adj, None, fit_intercept=True).fit(window=None, axis=1)

        return r

    
    @staticmethod
    def test_statistics(results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate test statistics from Fama-MacBeth regression results.
        """
        raise NotImplementedError("This method needs to be implemented.")