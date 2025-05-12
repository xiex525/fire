import pandas as pd
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
