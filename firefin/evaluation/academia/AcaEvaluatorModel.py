import typing
import pandas as pd
from ..eva_utils import compute_ic, ForwardReturns, QuantileReturns
from ...core.algorithm.regression import least_square, rolling_regression, BatchRegressionResult
from ...common.config import logger
from .anomaly_test import AnomalyTest
from .fama_macbeth import FamaMacBeth
from .portfolio_sort import PortfolioSort

class AcaEvaluatorModel:
    def __init__(self,
                 factor: pd.DataFrame,
                 forward_returns: ForwardReturns, 
                 return_adj: pd.DataFrame,
                 n_jobs: int = 10,
                 verbose: int = 0):
        """
        Parameters:
            factor: pd.DataFrame
                Factor exposure data (Time × Stock)
            forward_returns: dict[str, pd.DataFrame]
                A dictionary where each key is a holding period, and the value is a DataFrame of future returns (Time × Stock)
            return_adj: pd.DataFrame
                DataFrame of adjusted returns (Time × Stock)
            n_jobs: int
                Number of jobs to run in parallel
            verbose: int
                Verbosity level
        """

        self.factor = factor
        self.forward_returns = forward_returns
        self.return_adj = return_adj
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run_single_sort(self,
                        quantiles: int = 5,
                        value_weighted: bool = True,
                        return_stats: bool = False,
                        market_cap: pd.DataFrame = None,
                        get_quantile_sorts: bool = False):
        """
        Perform single-factor portfolio sorting to compute returns for each quantile group, 
        with optional return of statistics and quantile labels.

        Parameters:
            quantiles: int
                Number of quantile groups (e.g., 5 for quintile sorting)
            value_weighted: bool
                Whether to use value-weighted portfolios; False indicates equal-weighted portfolios
            return_stats: bool
                Whether to compute and return statistics (mean, t-stat, p-value, etc.) for the H-L portfolio
            market_cap: pd.DataFrame
                Market capitalization data, with the same dimensions as the factor; required if value_weighted is True
            get_quantile_sorts: bool
                Whether to return the quantile label assigned to each stock

        Returns:
            If return_stats is True:
                Tuple[QuantileReturns, dict] → (portfolio returns, dictionary of statistics)
            Otherwise:
                QuantileReturns
        """

        if value_weighted and market_cap is None:
            raise ValueError("You must provide market_cap when value_weighted=True.")

        portfolio_returns = PortfolioSort.single_sort(
            factor=self.factor,
            forward_returns=self.forward_returns,
            market_cap=market_cap,
            quantiles=quantiles,
            value_weighted=value_weighted,
            get_quantile_sorts=get_quantile_sorts
        )

        if return_stats:
            stats = PortfolioSort.get_statistics(portfolio_returns, quantiles)
            return portfolio_returns, stats

        return portfolio_returns

    def run_fama_macbeth(self,
                         window: int = 252,
                         return_stats: bool = False):
        """
        Perform Fama-MacBeth two-stage cross-sectional regression estimation.

        Parameters:
            window: int
                Rolling window size for the first-stage regressions (default is 252, i.e., one year)
            return_stats: bool
                Whether to return t-statistics and significance test results

        Returns:
            If return_stats is True:
                Tuple[RegressionResult, dict] → (regression results, statistics)
            Otherwise:
                RegressionResult
        """

        results = FamaMacBeth.run_regression(self.factor, self.return_adj, window=window, n_jobs=self.n_jobs, verbose=self.verbose)
        if return_stats:
            stats = FamaMacBeth.test_statistics(results)
            return results, stats
        return results
        
    def run_ic(self, method: str = "pearson") -> pd.DataFrame:
        """
        Compute the Information Coefficient (IC) between the factor and future returns.

        Parameters:
            method: str
                Correlation method to use; options are: 'pearson', 'spearman', 'kendall'
    
        Returns:
            pd.DataFrame
                IC values for each period
        """

        return compute_ic(self.factor, self.forward_returns, method=method)

    def run_regression(self, rolling: bool = False, window: int = 60, fit_intercept: bool = True) -> BatchRegressionResult | dict:
        """
        Run either static or rolling regression of returns on factor exposures.

        Parameters
        ----------
        rolling : bool, optional
            Whether to perform rolling regression, by default False.
        window : int, optional
            Rolling window size (only used if rolling=True), by default 60.
        fit_intercept : bool, optional
            Whether to include an intercept in the regression, by default True.

        Returns
        -------
        BatchRegressionResult | dict
            Regression result object (static) or a dictionary of rolling results.
        """
        if rolling:
            # Use rolling_regression function
            result = rolling_regression(x=self.factor, y=self.return_adj, window=window, fit_intercept=fit_intercept, n_jobs=self.n_jobs, verbose=self.verbose)
        else:
            # Time-by-time regression using least_square
            from collections import defaultdict
            results = defaultdict(list)
            for t in self.factor.index:
                x_t = self.factor.loc[t]
                y_t = self.return_adj.loc[t]
                if x_t.isnull().any() or y_t.isnull().any():
                    continue
                reg_result = least_square(x=x_t, y=y_t, fit_intercept=fit_intercept)
                results['alpha'].append(reg_result.alpha)
                results['beta'].append(reg_result.beta)
                results['r2'].append(reg_result.r2)
                results['r2_adj'].append(reg_result.r2_adj)
                results['residuals'].append(reg_result.residuals)
            result = BatchRegressionResult(alpha=results['alpha'], beta=results['beta'], r2=results['r2'], r2_adj=results['r2_adj'], residuals=results['residuals'])
        return result
        
    def run_anomaly_test(self,
                         portfolio_returns: QuantileReturns,
                         cov_type: typing.Optional[str] = None,
                         cov_kwds: typing.Optional[dict] = None,
                         return_stats: bool = False):
        """
        Perform anomaly test by regressing portfolio returns on a factor model.

        Parameters:
            return_stats : bool
                Whether to return regression statistics summary.

        Returns:
            If return_stats is True:
                Tuple[AnomalyTest, pd.DataFrame]
            Else:
                AnomalyTest
        """
        mkt_ret = pd.DataFrame(self.return_adj.mean(axis=1))
        tester = AnomalyTest(portfolio_returns= portfolio_returns, factor_model=mkt_ret)
        
        if return_stats:
            summary = tester.fit(cov_type=cov_type, cov_kwds=cov_kwds).test_statistics()
            return summary
        return tester


    def run_all(self) -> dict:
        """
        Run all available evaluation methods and return results in a dictionary.

        Returns
        -------
        dict
            A dictionary containing the results of all evaluation methods.
        """
        results = {}
        #Single Sort
        logger.info("Running Single Sort")
        results['single_sort_res'], results['single_sort_stat'] = self.run_single_sort(
            quantiles=5,
            value_weighted=False,
            return_stats=True
        )
        logger.info("Single Sort Completed")
        #Fama-MacBeth Regression
        logger.info("Running Fama-MacBeth Regression")
        results['fama_macbeth_res'], results['fama_macbeth_stat']= self.run_fama_macbeth(
            window=252,
            return_stats=True
        )
        logger.info("Fama-MacBeth Regression Completed")
        # IC
        logger.info("Running IC")
        results['information_coefficient'] = self.run_ic(method="pearson")
        logger.info("IC Completed")

        # Static Regression
        logger.info("Running Static Regression")
        results['regression'] = self.run_regression(rolling=False, fit_intercept=True)
        logger.info("Static Regression Completed")
            
        # Anomaly Test
        logger.info("Running Anomaly Test")
        for k, v in results['single_sort_res'].items():
            results['anomaly_stat'] = {k:self.run_anomaly_test(portfolio_returns= pd.DataFrame(v.iloc[:,-1]), return_stats= True)}
        logger.info("Anomaly Test Completed")

        return results