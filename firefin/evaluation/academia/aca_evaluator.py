import typing
import pandas as pd
from ...core.plot import plots
from ..eva_utils import ForwardReturns, compute_ic 
from .portfolio_sort import PortfolioSort
from .fama_macbeth import FamaMacBeth
from .MSR_Test import MSRTest
from ...core.algorithm.regression import least_square, rolling_regression, BatchRegressionResult

class AcaEvaluator:
    def __init__(self, factor: pd.DataFrame, forward_returns: ForwardReturns):
        """
        Parameters:
            factor: pd.DataFrame
                Factor exposure data (Time × Stock)
            forward_returns: dict[str, pd.DataFrame]
                A dictionary where each key is a holding period, and the value is a DataFrame of future returns (Time × Stock)
        """

        self.factor = factor
        self.forward_returns = forward_returns

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

    def run_double_sort(self,
                        factor2: pd.DataFrame,
                        quantiles: tuple = (5, 5),
                        dependent: bool = False,
                        value_weighted: bool = True,
                        market_cap: pd.DataFrame = None,
                        get_quantile_sorts: bool = False):
        """
        Perform double-factor sorting by jointly grouping assets based on factor1 and factor2, and calculate returns.

        Parameters:
            factor2: pd.DataFrame
                The second factor, must have the same dimensions as self.factor
            quantiles: Tuple[int, int]
                Number of quantile groups for the primary and secondary factors (e.g., (5, 5))
            dependent: bool
                Whether to use conditional (nested) sorting
            value_weighted: bool
                Whether to use value-weighted portfolios
            market_cap: pd.DataFrame
                Market capitalization data, same dimensions as the factors; required if value_weighted is True
            get_quantile_sorts: bool
                Whether to return portfolio labels (i.e., the group each stock belongs to)

        Returns:
            QuantileReturns or dict[str, pd.DataFrame] (if get_quantile_sorts is True)
        """

        if value_weighted and market_cap is None:
            raise ValueError("You must provide market_cap when value_weighted=True.")

        return PortfolioSort.double_sort(
            factor1=self.factor,
            factor2=factor2,
            forward_returns=self.forward_returns,
            market_cap=market_cap,
            quantiles=quantiles,
            dependent=dependent,
            value_weighted=value_weighted,
            get_quantile_sorts=get_quantile_sorts
        )

    def run_fama_macbeth(self,
                         return_adj: pd.DataFrame,
                         window: int = 252,
                         return_stats: bool = False):
        """
        Perform Fama-MacBeth two-stage cross-sectional regression estimation.

        Parameters:
            return_adj: pd.DataFrame
                Adjusted returns for each period (e.g., monthly returns)
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

        results = FamaMacBeth.run_regression(self.factor, return_adj, window=window)
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

    def run_msr_test(self, model_a, model_b, regularize=True):
        """
        Compare the Maximum Sharpe Ratios of two factor models using a Z-test.
        Args:
            model_a (pd.DataFrame): T × K matrix of factor returns for model A.
            model_b (pd.DataFrame): T × K matrix of factor returns for model B.
            regularize_covariance (bool): If True, regularize the covariance matrix.
        Returns:
            dict: {
                'msr_a': float,  # MSR of model A
                'msr_b': float,  # MSR of model B
                'test_stat': float,  # Z-statistic
                'p_value': float  # two-sided p-value
            }
        """
        return MSRTest.run_msr_comparison(model_a, model_b, regularize_covariance=True)

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
            result = rolling_regression(x=self.factor, y=self.forward_returns, window=window, fit_intercept=fit_intercept)
        else:
            # Time-by-time regression using least_square
            from collections import defaultdict
            results = defaultdict(list)
            for t in self.factor.index:
                x_t = self.factor.loc[t]
                y_t = self.forward_returns.loc[t]
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

    
    
    
    
    
    
        
    def get_grs_test(self, test_portfolios: pd.DataFrame, plot=True) -> dict:
        """
        对测试组合进行 GRS 检验，判断资产定价模型能否整体解释这些组合的收益（所有 alpha 是否为 0）
        输入：
            test_portfolios: 测试组合的时间序列收益（行是时间，列是组合名）
            plot: 是否绘图
    
        输出：
            dict，包括：
                - grs_stat: GRS 检验统计量
                - p_value: GRS 检验的 p 值（是否拒绝所有 alpha 为 0）
                - alphas: 各组合的 alpha（Series）
                - t_stats: alpha 的 t 值（Series）
                - residual_cov: 残差协方差矩阵（DataFrame）
                - betas: 回归系数矩阵（DataFrame）
    
        工具函数接口预设：
        def run_grs_test(factor: pd.DataFrame,
                         test_portfolios: pd.DataFrame) -> dict:
            # 返回值格式如下：
            # {
            #   "grs_stat": float,
            #   "p_value": float,
            #   "alphas": Series,
            #   "t_stats": Series,
            #   "residual_cov": DataFrame,
            #   "betas": DataFrame
            # }
    
        def plot_grs_result(alphas: pd.Series, t_stats: pd.Series, grs_stat: float, p_value: float):
            # 可选绘图函数：展示 alpha/t 值分布及整体显著性
        """
        cache_key = ("grs_test",)
        if cache_key not in self._result:
            result = run_grs_test(self.factor, test_portfolios)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_grs_result(
                result["alphas"], result["t_stats"],
                result["grs_stat"], result["p_value"]
            )
        return result

    def get_hj_distance_test(self, test_portfolios: pd.DataFrame, plot=True) -> dict:
        """
        对指定测试组合，计算该因子模型的 Hansen–Jagannathan 距离，用于衡量模型定价误差
        输入：
            test_portfolios: 测试组合收益（T × N）
            plot: 是否绘图
    
        输出：
            dict，包括：
                - hj_distance: HJ 距离
                - hj_std: 距离的标准误
                - t_stat: HJ 距离是否显著大于 0 的 t 值
                - p_value: 检验 p 值
                - alpha: 各组合 alpha（定价误差）
                - residual_cov: 残差协方差矩阵
                - betas: 模型回归系数
    
        工具函数接口预设：
        def run_hj_distance(factor: pd.DataFrame, 
                            test_portfolios: pd.DataFrame) -> dict:
            # 返回结构：
            # {
            #   "hj_distance": float,
            #   "hj_std": float,
            #   "t_stat": float,
            #   "p_value": float,
            #   "alpha": Series,
            #   "betas": DataFrame,
            #   "residual_cov": DataFrame
            # }
    
        def plot_hj_distance_result(hj_distance: float, t_stat: float):
            # 可视化：展示距离大小及显著性
        """
        cache_key = ("hj_distance_test",)
        if cache_key not in self._result:
            result = run_hj_distance(self.factor, test_portfolios)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_hj_distance_result(result["hj_distance"], result["t_stat"])
        return result

    def compare_model_alphas(self, 
                         models: dict[str, pd.DataFrame], 
                         test_portfolios: pd.DataFrame, 
                         plot=True) -> dict:
        """
        比较不同模型在相同测试组合上的 alpha（截距项）大小与显著性
        输入：
            models: dict[str, DataFrame]，键为模型名，值为该模型的因子数据（T × K）
            test_portfolios: 测试组合收益（T × N）
            plot: 是否绘图
    
        输出：
            dict，键为模型名，值为该模型的回归结果：
                - alpha: 每个组合的 alpha（Series）
                - t_stat: alpha 的 t 值（Series）
                - mean_abs_alpha: 平均 |alpha|
                - mean_abs_t: 平均 |t|
    
        工具函数接口预设：
        def run_alpha_estimation(factor: pd.DataFrame, test_portfolios: pd.DataFrame) -> dict:
            # 返回结构：
            # {
            #   "alpha": Series,
            #   "t_stat": Series
            # }
    
        def plot_model_alpha_comparison(results: dict):
            # 可视化不同模型下 mean |alpha| 与 mean |t|
        """
        cache_key = ("alpha_comparison", tuple(models.keys()))
        if cache_key not in self._result:
            comparison_result = {}
    
            for model_name, factor_df in models.items():
                result = run_alpha_estimation(factor_df, test_portfolios)
    
                alpha = result["alpha"]
                t_stat = result["t_stat"]
                mean_abs_alpha = alpha.abs().mean()
                mean_abs_t = t_stat.abs().mean()
    
                comparison_result[model_name] = {
                    "alpha": alpha,
                    "t_stat": t_stat,
                    "mean_abs_alpha": mean_abs_alpha,
                    "mean_abs_t": mean_abs_t
                }
    
            self._result[cache_key] = comparison_result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_model_alpha_comparison(result)
        return result

    def run_horse_race_regression(self, 
                                   candidate_factors: dict[str, pd.DataFrame], 
                                   forward_return_key: str, 
                                   date: str = None, 
                                   plot=True) -> dict:
        """
        同时比较多个因子在控制其他因子的条件下的边际解释力（截面回归）
        输入：
            candidate_factors: dict[str, DataFrame]，键为因子名，值为因子值（T × 股票）
            forward_return_key: 使用哪一组 forward return（例如 '1M', '3M'）
            date: 若指定，仅对该日期做单期截面回归；若 None，则对所有期进行多期平均
            plot: 是否绘图
    
        输出：
            dict，包括：
                - coefs: 单期为 Series；多期为 DataFrame（时间 × 因子）
                - mean_coef: 多期平均系数（Series，仅在多期模式）
                - t_stat: 多期系数的 t 值（Series，仅在多期模式）
                - p_value: 多期系数的 p 值（Series，仅在多期模式）
    
        工具函数接口预设：
        def run_cross_sectional_horse_race(factors: dict[str, pd.DataFrame], 
                                           forward_returns: pd.DataFrame,
                                           date: str = None) -> dict:
            # 若 date 非 None，返回：
            #   { "coefs": Series }
            # 否则返回：
            #   {
            #     "coefs": DataFrame(时间 × 因子),
            #     "mean_coef": Series,
            #     "t_stat": Series,
            #     "p_value": Series
            #   }
    
        def plot_horse_race_t_stats(t_stats: pd.Series):
            # 可视化回归中各因子的 t 值比较
        """
        cache_key = ("horse_race", tuple(candidate_factors.keys()), date)
        if cache_key not in self._result:
            # 构造因子字典按截面对齐
            factors_aligned = {k: v.loc[self.factor.index] for k, v in candidate_factors.items()}
            target_returns = self.forward_returns[forward_return_key]
    
            result = run_cross_sectional_horse_race(factors_aligned, target_returns, date)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot and "t_stat" in result:
            plots.plot_horse_race_t_stats(result["t_stat"])
        return result

    def get_spanning_test(self, 
                          new_factor: pd.Series, 
                          base_model_factors: pd.DataFrame, 
                          plot=True) -> dict:
        """
        检验某个新因子是否可以由已有因子组合线性表示
        
        输入：
            new_factor: 新因子的时间序列（pd.Series, index 与 base_model_factors 对齐）
            base_model_factors: 原模型因子（DataFrame: 时间 × K）
            plot: 是否绘图
    
        输出：
            dict，包括：
                - r_squared: 新因子被原模型拟合的 R²（越高越可能冗余）
                - t_stat: 截距项 alpha 的 t 值（是否显著不为 0）
                - p_value: p 值（是否拒绝 H₀: new factor 是冗余的）
                - alpha: 截距项（若显著，则说明该因子含有原模型未能 span 的信息）
                - beta: 回归系数（新因子在 base_model 上的加载系数）
                - resid_std: 残差标准差
    
        工具函数接口预设：
        def run_spanning_test(new_factor: pd.Series, 
                              base_factors: pd.DataFrame) -> dict:
            # 返回结构：
            # {
            #   "alpha": float,
            #   "beta": Series,
            #   "r_squared": float,
            #   "t_stat": float,
            #   "p_value": float,
            #   "resid_std": float
            # }
    
        def plot_spanning_result(alpha: float, t_stat: float, r_squared: float):
            # 可视化截距的显著性与拟合效果
        """
        cache_key = ("spanning_test", new_factor.name, tuple(base_model_factors.columns))
        if cache_key not in self._result:
            result = run_spanning_test(new_factor, base_model_factors)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_spanning_result(result["alpha"], result["t_stat"], result["r_squared"])
        return result

    def run_subsample_analysis(self, 
                               method: typing.Literal["ic", "alpha", "quantile_returns"], 
                               split_dates: list[str], 
                               forward_return_key: str = None,
                               quantiles: int = 5,
                               plot=True) -> dict:
        """
        子样本稳健性检验：将全样本按时间切割为多个子样本，检验因子表现是否稳定
    
        输入：
            method: 分析方式
                - "ic"：计算每个子样本的平均信息系数及 t 值
                - "alpha"：计算每段时间对测试组合的回归 alpha
                - "quantile_returns"：比较分层收益
            split_dates: 时间断点列表，例如 ["2005-01-01", "2015-01-01"] → 分三段
            forward_return_key: 用于 IC 或 quantile analysis 的 forward return 名称
            quantiles: 用于分组排序的组数
            plot: 是否绘图
    
        输出：
            dict，键为区间描述，值为分析结果：
                - 如果是 IC → { "mean_ic", "t_stat", "p_value" }
                - 如果是 alpha → { "alphas", "t_stats", "mean_abs_alpha" }
                - 如果是 quantile_returns → 分组收益结构体
    
        工具函数接口预设：
        def run_ic_on_subsample(factor, forward_return, date_range) -> dict:
            # 返回 { "mean_ic": float, "t_stat": float, "p_value": float }
    
        def run_alpha_on_subsample(factor, forward_return, portfolios, date_range) -> dict:
            # 返回 { "alphas": Series, "t_stats": Series, "mean_abs_alpha": float }
    
        def run_quantile_returns_on_subsample(factor, forward_return, date_range, quantiles) -> dict:
            # 返回排序结果，如 quantile_returns 表或图表数据结构
    
        def plot_subsample_comparison(results: dict, method: str):
            # 根据 method 显示对比图
        """
        cache_key = ("subsample_analysis", method, tuple(split_dates))
        if cache_key not in self._result:
            all_dates = [self.factor.index.min()] + split_dates + [self.factor.index.max()]
            results = {}
    
            for i in range(len(all_dates) - 1):
                start, end = all_dates[i], all_dates[i + 1]
                date_range = (start, end)
                label = f"{start[:10]} ~ {end[:10]}"
    
                if method == "ic":
                    forward = self.forward_returns[forward_return_key]
                    results[label] = run_ic_on_subsample(self.factor, forward, date_range)
    
                elif method == "alpha":
                    # portfolios 应该预先设定在 self 或作为参数传入
                    forward = self.forward_returns[forward_return_key]
                    results[label] = run_alpha_on_subsample(self.factor, forward, self.test_portfolios, date_range)
    
                elif method == "quantile_returns":
                    forward = self.forward_returns[forward_return_key]
                    results[label] = run_quantile_returns_on_subsample(self.factor, forward, date_range, quantiles)
    
                else:
                    raise ValueError(f"Unknown method: {method}")
    
            self._result[cache_key] = results
    
        result = self._result[cache_key]
        if plot:
            plots.plot_subsample_comparison(result, method)
        return result
        
    def compute_vif(self, factors: pd.DataFrame, plot=True) -> dict:
        """
        计算因子之间的方差膨胀因子（VIF），用于诊断多重共线性问题    
        输入：
            factors: pd.DataFrame，T × K 的因子矩阵（行是时间，列是因子）
            plot: 是否绘图
    
        输出：
            dict，包括：
                - vif: pd.Series，每个因子的 VIF 值
                - max_vif: 最大的 VIF（标识最严重的共线性）
                - mean_vif: 平均 VIF
    
        工具函数接口预设：
        def calculate_vif(factors: pd.DataFrame) -> pd.Series:
            # 返回每个因子的 VIF 值（Series）
    
        def plot_vif_bar(vif_series: pd.Series):
            # 可视化 VIF 条形图
        """
        cache_key = ("vif", tuple(factors.columns))
        if cache_key not in self._result:
            vif_series = calculate_vif(factors)
            result = {
                "vif": vif_series,
                "max_vif": vif_series.max(),
                "mean_vif": vif_series.mean()
            }
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_vif_bar(result["vif"])
        return result
