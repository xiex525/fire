"""
Portfolio Sort Implementation for Academic Research
---------------------------------------------------
This module provides a class for performing single and double portfolio sorts
based on characteristics, market capitalization, and returns. The implementation
focuses on clarity, documentation, and best practices for financial research.
"""

import typing
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from ..eva_utils import factor_to_quantile
from ..eva_utils import _compute_quantile_df_df, _compute_weighted_quantile_df
from ..eva_utils import ForwardReturns, QuantileReturns

StatisticResults = typing.NewType("StatisticResults", dict[str, pd.DataFrame])

class PortfolioSort:
    """
    Class to perform single and double portfolio sorts based on characteristics.
    """

    @staticmethod
    def single_sort(
        factor: pd.DataFrame,
        forward_returns: ForwardReturns,
        market_cap: pd.DataFrame,
        quantiles: int,
        min_assets: int = 10,
        value_weighted: bool = True,
        get_series: bool = False,
        get_quantile_sorts: bool = False,
        get_tstat: bool = False,
        char_lag: int = -1,
    ) -> typing.Union[pd.DataFrame, np.ndarray]:
        """
        Perform single portfolio sort based on characteristic and create long-short portfolio.
        
        Args:
            factor: TxN DataFrame of characteristic exposures
            forward_returns: TxN DataFrame of returns
            market_cap: TxN DataFrame of market capitalizations
            quantiles: number of quantiles
            min_assets: Minimum required assets per portfolio
            value_weighted: Use market cap weighting (True) or equal weighting (False)
            get_series: Return time series instead of averages
            get_quantile_sorts: Return portfolio assignments
            get_tstat: Return t-statistics instead of p-values
            char_lag: Lag between characteristic and return calculation
        Returns:
            Portfolio returns and statistical results
        """
        # 1. DATA PREPARATION
        # assume factor, forward_return, market_cap are aligned DataFrames in our case
        # 2. QUANTILE CALCULATIONS
        quantile_sorts = factor_to_quantile(factor, quantiles)

        # Early exit if quantile assignments requested
        if get_quantile_sorts:
            return quantile_sorts
        
        # 3. RETURN CALCULATIONS
        if value_weighted:
            portfolio_returns = QuantileReturns ({
                period: _compute_weighted_quantile_df(quantile_sorts, period_returns, market_cap,quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })
        else:
            # equal weighted
            portfolio_returns = QuantileReturns ({
                period: _compute_quantile_df_df(quantile_sorts, period_returns, quantiles=quantiles)
                for period, period_returns in forward_returns.items()
                })

        # 4. HEDGE PORTFOLIO (High-Low)
        for period, _ in forward_returns.items():
            portfolio_returns[period]["H-L"] = (
                portfolio_returns[period][quantiles] - portfolio_returns[period][1]
            )
    
        return portfolio_returns

    @staticmethod
    def get_statistics(result: QuantileReturns, quantiles: int) -> StatisticResults:
        """
        Compute statistical results for single portfolio sort.

        TODO: 
            1. Add more statistics
            2. plot the results
        """        
        # T-Test for all periods
        # periods * (quantiles + H-L)
        t_stats = np.empty((len(result), quantiles + 1), dtype=float)
        p_values = np.empty((len(result), quantiles + 1), dtype=float)
        mean_returns = np.empty((len(result), quantiles + 1), dtype=float)
                                
        for n, (_, period_returns) in enumerate(result.items()):
            # T-Test for all periods
            t_stats[n], p_values[n] = np.apply_along_axis(
                ttest_1samp,
                0,
                period_returns,
                popmean=0,
                nan_policy='omit'
            )
            # other statistics can be added here
            mean_returns[n] = np.nanmean(period_returns, axis=0)

        return StatisticResults({'t_stats': pd.DataFrame(t_stats, index=result.keys(), columns=period_returns.columns),
                'p_values': pd.DataFrame(p_values, index=result.keys(), columns=period_returns.columns),
                'mean_returns': pd.DataFrame(mean_returns, index=result.keys(), columns=period_returns.columns)})