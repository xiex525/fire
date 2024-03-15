# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import typing

import numpy as np
import pandas as pd

__all__ = [
    "compute_forward_returns",
    "compute_ic",
    "factor_to_quantile",
    "compute_quantile_returns",
]

PeriodType = typing.NewType("PeriodType", int)
ForwardReturns = typing.NewType("ForwardReturns", dict[PeriodType, pd.DataFrame])
IC = typing.NewType("IC", pd.DataFrame)
QuantileReturns = typing.NewType("QuantileReturns", dict[PeriodType, pd.DataFrame])


def compute_forward_returns(price: pd.DataFrame, periods: list[PeriodType]) -> ForwardReturns:
    forward_returns_dict = {}

    returns: pd.DataFrame = np.log(price).shift(-1) - np.log(price)

    for period in sorted(periods):
        if period == 1:
            forward_returns_dict[period] = returns
            continue

        log_period_returns = returns.rolling(period).sum().shift(1 - period)
        period_returns: pd.DataFrame = np.exp(log_period_returns) - 1
        forward_returns_dict[period] = period_returns
    return ForwardReturns(forward_returns_dict)


def _compute_ic_df_df(
    a: pd.DataFrame, b: pd.DataFrame, method: typing.Literal["pearson", "kendall", "spearman"]
) -> pd.Series:
    return a.corrwith(b, axis=1, method=method)


def compute_ic(
    factor: pd.DataFrame, forward_returns: ForwardReturns, method: typing.Literal["pearson", "kendall", "spearman"]
) -> IC:
    """
    Compute IC (Information Coefficient) for the factor and forward returns, which is the correlation between the
    factor and the forward returns.

    Parameters
    ----------
    factor: pd.DataFrame
    forward_returns: ForwardReturns
    method: str
        default "pearson"

    Returns
    -------
    IC
        a dataframe of IC values for each period in columns.

    """
    factor = factor[np.isfinite(factor)]
    return IC(
        pd.DataFrame(
            {
                period: _compute_ic_df_df(factor, period_returns, method=method)
                for period, period_returns in forward_returns.items()
            }
        )
    )


def factor_to_quantile(factor: pd.DataFrame, quantiles: int = 5) -> pd.DataFrame:
    """
    Convert factor to quantile row-wise. The result will always have quantile values ranging from `quantiles` down
    to 1 continuously (if only 1 group, it'll be `quantiles`).

    Parameters
    ----------
    factor: pd.DataFrame
    quantiles: int
        default 5

    Returns
    -------
    pd.DataFrame
        a dataframe of quantile values.

    """
    quantile_values = np.arange(1, quantiles + 1)

    def _row_to_quantile(row):
        finite = np.isfinite(row)
        if finite.any():
            tmp: pd.Series = pd.qcut(row[finite], quantiles, labels=False, duplicates="drop")
            # rearrange values from `q` to 1
            # this makes sure that the quantile values are generally continuous,
            # and we always have a group of long portfolio of `q`
            old_values = tmp.unique()
            old_values.sort()
            new_values = quantile_values[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                tmp.replace(old_values, new_values, inplace=True)
            row = row.copy()
            row[finite] = tmp
            return row
        else:
            return row

    return factor.apply(_row_to_quantile)


def _compute_quantile_df_df(qt: pd.DataFrame, fr: pd.DataFrame, quantiles: int = 5):
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row) in zip(fr.iterrows(), qt.iterrows()):
        result[dt] = fr_row.groupby(qt_row).mean()
    result = pd.DataFrame(result).T
    result = result.reindex(columns=np.arange(1, quantiles + 1), copy=False)
    return result


def compute_quantile_returns(
    factor: pd.DataFrame, forward_returns: ForwardReturns, quantiles: int = 5
) -> QuantileReturns:
    """
    Compute quantile returns. Factor will be converted to quantiles using `factor_to_quantile`. Then, for each period
    in forward_returns, the period returns will be grouped row-wise by quantiles and averaged.

    Parameters
    ----------
    factor: pd.DataFrame
    forward_returns: ForwardReturns
    quantiles: int
        default 5

    Returns
    -------
    QuantileReturns
        a dictionary of period returns for each quantile. The quantile returns are dataframe with index as date and
        columns as quantiles.

    """
    factor_as_quantile = factor_to_quantile(factor, quantiles=quantiles)
    return QuantileReturns(
        {
            period: _compute_quantile_df_df(factor_as_quantile, period_returns, quantiles=quantiles)
            for period, period_returns in forward_returns.items()
        }
    )
