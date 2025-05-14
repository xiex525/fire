import typing
import pandas as pd
from ...core.plot import plots
from ..eva_utils import ForwardReturns
from .portfolio_sort import PortfolioSort
from .fama_macbeth import FamaMacBeth


class AcaEvaluator:
    def __init__(self, factor: pd.DataFrame, forward_returns: ForwardReturns):
        """
        参数:
            factor: pd.DataFrame
                因子暴露数据，维度为 (时间 × 股票)
            forward_returns: dict[str, pd.DataFrame]
                每个持有期对应未来收益率（时间 × 股票）DataFrame
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
        执行单因子分组排序，计算各分组收益，可选返回统计量与分组标签。
        参数:
            quantiles: int
                分组数（如 5 表示五分位分组）
            value_weighted: bool
                是否使用市值加权；False 表示等权组合
            return_stats: bool
                是否计算并返回 H-L 投资组合的统计量（均值、t 值、p 值等）
            market_cap: pd.DataFrame
                市值数据，维度与因子相同；在 value_weighted=True 时必需
            get_quantile_sorts: bool
                是否返回每只股票所属的分位编号

        返回:
            如果 return_stats 为 True:
                Tuple[QuantileReturns, dict] → (组合收益，统计量字典)
            否则:
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
        执行双因子排序，将资产按 factor1、factor2 联合分组并计算收益。
        参数:
            factor2: pd.DataFrame
                第二个因子，维度应与 self.factor 相同
            quantiles: Tuple[int, int]
                主因子和次因子的分组数量（如 (5, 5)）
            dependent: bool
                是否使用条件排序（nested sort）
            value_weighted: bool
                是否使用市值加权
            market_cap: pd.DataFrame
                市值数据，维度与因子相同；在 value_weighted=True 时必需
            get_quantile_sorts: bool
                是否返回组合标签（即每只股票所在的分组）

        返回:
            QuantileReturns 或 dict[str, pd.DataFrame]（如果 get_quantile_sorts 为 True）
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
        执行 Fama-MacBeth 双阶段横截面回归估计。
        参数:
            return_adj: pd.DataFrame
                每期的收益率（如月度收益）
            window: int
                第一阶段回归的滚动窗口大小（默认为 252，即一年）
            return_stats: bool
                是否返回 t 统计量和显著性检验结果

        返回:
            如果 return_stats 为 True:
                Tuple[RegressionResult, dict] → （回归结果，统计量）
            否则:
                RegressionResult
        """
        results = FamaMacBeth.run_regression(self.factor, return_adj, window=window)
        if return_stats:
            stats = FamaMacBeth.test_statistics(results)
            return results, stats
        return results
        
    def run_ic(self, method: str = "pearson") -> pd.DataFrame:
        """
        计算因子与未来收益之间的 IC（信息系数）
        参数:
            method: str
                使用的相关系数方法，可选：'pearson', 'spearman', 'kendall'
    
        返回:
            pd.DataFrame
                每期对应的 IC 值
        """
        from ..eva_utils import compute_ic 
        return compute_ic(self.factor, self.forward_returns, method=method)





    def get_time_series_regression(self, test_portfolios: pd.DataFrame, plot=True) -> dict:
        """
        对每个测试组合进行时间序列回归，分析模型能否解释该组合收益（即检验 alpha 是否显著）
        输入：
            test_portfolios: 待解释的组合收益（行是时间，列是组合名）
            plot: 是否绘图
        输出：
            dict，包括：
                - alphas: 每个组合的 alpha（Series）
                - t_stats: alpha 的 t 统计量（Series）
                - p_values: alpha 的 p 值（Series）
                - betas: 每个组合对应的因子暴露（DataFrame）
    
        工具函数接口预设：
        def run_time_series_regression(factor: pd.DataFrame, 
                                        test_portfolios: pd.DataFrame) -> dict:
            # 返回结果结构：
            # {
            #   "alphas": Series,
            #   "t_stats": Series,
            #   "p_values": Series,
            #   "betas": DataFrame
            # }
    
        def plot_alpha_distribution(alphas: pd.Series):
            # 绘制 alpha 的分布图
    
        def plot_alpha_significance(t_stats: pd.Series):
            # 可以绘制 t 值条形图或标记显著组合
        """
        cache_key = ("time_series_regression",)
        if cache_key not in self._result:
            result = run_time_series_regression(self.factor, test_portfolios)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_alpha_distribution(result["alphas"])
            plots.plot_alpha_significance(result["t_stats"])
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

    def get_sharpe_ratio_test(self, 
                          model_a_factors: pd.DataFrame, 
                          model_b_factors: pd.DataFrame, 
                          plot=True) -> dict:
        """
        比较两个因子模型的最大 Sharpe 比率是否显著不同
        输入：
            model_a_factors: 模型 A 的因子收益
            model_b_factors: 模型 B 的因子收益
            plot: 是否绘图
        输出：
            dict，包括：
                - sr_a: 模型 A 的最大 Sharpe 比率
                - sr_b: 模型 B 的最大 Sharpe 比率
                - test_stat: 检验统计量（差异显著性）
                - p_value: p 值（是否拒绝 SR 相等的原假设）
    
        工具函数接口预设：
        def run_sharpe_ratio_test(model_a: pd.DataFrame, model_b: pd.DataFrame) -> dict:
            # 返回值结构：
            # {
            #   "sr_a": float,
            #   "sr_b": float,
            #   "test_stat": float,
            #   "p_value": float
            # }
    
        def plot_sharpe_comparison(sr_a: float, sr_b: float):
            # 可视化 Sharpe 比率对比
        """
        cache_key = ("sharpe_ratio_test",)
        if cache_key not in self._result:
            result = run_sharpe_ratio_test(model_a_factors, model_b_factors)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_sharpe_comparison(result["sr_a"], result["sr_b"])
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

    def run_rolling_regression(self, 
                               test_portfolios: pd.DataFrame, 
                               window: int = 60, 
                               min_obs: int = 36,
                               plot=True) -> dict:
        """
        对每个测试组合做滚动窗口时间序列回归，观察 alpha 和 beta 是否随时间变化
    
        输入：
            test_portfolios: DataFrame，组合收益（时间×组合）
            window: 滚动窗口长度（单位：月，60表示5年）
            min_obs: 最小有效样本数要求（窗口内数据不足将跳过）
            plot: 是否绘图
    
        输出：
            dict，包括：
                - rolling_alpha: DataFrame（时间 × 组合）
                - rolling_tstat: DataFrame（时间 × 组合）
                - rolling_beta: dict[str → DataFrame]，每个因子的时间序列系数
    
        工具函数接口预设：
        def run_rolling_ts_regression(factor: pd.DataFrame,
                                       test_portfolios: pd.DataFrame,
                                       window: int,
                                       min_obs: int) -> dict:
            # 返回结构：
            # {
            #   "rolling_alpha": DataFrame(时间 × 组合),
            #   "rolling_tstat": DataFrame(时间 × 组合),
            #   "rolling_beta": dict[str, DataFrame]（因子名 × 时间 × 组合）
            # }
    
        def plot_rolling_alpha_series(rolling_alpha: pd.DataFrame):
            # 可视化 alpha 随时间变化（可以选取几个组合）
    
        def plot_rolling_beta_series(rolling_beta: dict[str, pd.DataFrame]):
            # 可视化 beta 随时间的稳定性（可选因子 × 组合）
        """
        cache_key = ("rolling_regression", window, min_obs)
        if cache_key not in self._result:
            result = run_rolling_ts_regression(self.factor, test_portfolios, window, min_obs)
            self._result[cache_key] = result
    
        result = self._result[cache_key]
        if plot:
            plots.plot_rolling_alpha_series(result["rolling_alpha"])
            plots.plot_rolling_beta_series(result["rolling_beta"])
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
