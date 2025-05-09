# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/fire/blob/master/NOTICE.txt

# TODO: Move some common algorithms to fire/core/algorithm/

import typing

import numpy as np
import pandas as pd

__all__ = [
    "compute_forward_returns",
    "compute_ic",
    "factor_to_quantile",
    "factor_to_quantile_dependent_double_sort",
    "compute_quantile_returns",
    "_compute_weighted_quantile_df",
    "_compute_quantile_df",
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

def factor_to_quantile_dependent_double_sort(primary_factor: pd.DataFrame, secondary_factor: pd.DataFrame, quantiles: typing.Tuple[int, int]):
    """
    Perform dependent double sorting on two factors.

    Parameters:
    ------------
    primary_factor : pd.DataFrame
        The primary factor used for initial sorting.
    secondary_factor : pd.DataFrame
        The secondary factor used for sorting within each group defined by the primary factor.
    quantiles : tuple of int
       A tuple containing the number of quantiles for the primary and secondary factors respectively.
    
    Returns:
    --------
    quantile_sorts : pd.DataFrame
       A DataFrame where each entry represents the quantile assignment for the secondary factor within the group defined by the primary factor.
    
    TODO: numba jit acceleration
    """
    quantile_values_p = np.arange(1, quantiles[0] + 1)
    quantile_values_s = np.arange(1, quantiles[1] + 1)

    def _row_to_quantile(row_p, row_s):
        finite_p = np.isfinite(row_p)
        finite_s = np.isfinite(row_s)

        if finite_p.any() or finite_s.any():
            # Sort by primary factor first
            temp_p : pd.Series = pd.qcut(row_p[finite_p], quantiles[0], labels=False, duplicates='drop') 
            old_values = temp_p.unique()
            old_values.sort()
            new_values = quantile_values_p[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                temp_p.replace(old_values, new_values, inplace=True)

            # Sort by secondary factor within each primary quantile
            temp_s = pd.Series(np.zeros_like(row_p), index=row_p.index, dtype=int)
            temp_s[~finite_p | ~finite_s] = np.nan

            for q in quantile_values_p:
                mask = temp_p == q
                if mask.any():
                    # nan + nan, nan + int -> nan, int + nan -> nan, int + int -> int
                    temp_s[mask] = pd.qcut(row_s[finite_s & mask], quantiles[1], labels=False, duplicates='drop')
                else:
                    temp_s[mask] = np.nan
            
            old_values = temp_s.unique()
            old_values.sort()
            new_values = quantile_values_s[-len(old_values) :]
            if not np.array_equal(old_values, new_values):
                temp_s.replace(old_values, new_values, inplace=True)
            
            return temp_p.astype(str) + "_" + temp_s.astype(str)
        else:
            return pd.Series(index=row_p.index, dtype=str)

    result = pd.DataFrame(index=primary_factor.index, columns=primary_factor.columns)
    # apply the function to each row both of the factors
    for (i, row_p), (_, row_s) in zip(primary_factor.iterrows(), secondary_factor.iterrows()):
        result.loc[i] = _row_to_quantile(row_p, row_s)

    return result

def _compute_quantile_df(qt: pd.DataFrame, fr: pd.DataFrame, reindex=True, quantiles: int = 5):
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row) in zip(fr.iterrows(), qt.iterrows()):
        result[dt] = fr_row.groupby(qt_row).mean()
    result = pd.DataFrame(result).T
    if reindex:
        return result.reindex(columns=np.arange(1, quantiles + 1), copy=False)
    return result

def _compute_weighted_quantile_df(qt: pd.DataFrame, fr: pd.DataFrame, wt: pd.DataFrame, reindex= True, quantiles: int = 5):
    # assume aligned
    result = {}
    for (dt, fr_row), (_, qt_row), (_, wt_row) in zip(fr.iterrows(), qt.iterrows(), wt.iterrows()):
        _wt_row = wt_row.groupby(qt_row).transform(lambda x: x / x.sum())
        result[dt] = (fr_row * _wt_row).groupby(qt_row).sum()
    result = pd.DataFrame(result).T
    if reindex:
        return result.reindex(columns=np.arange(1, quantiles + 1), copy=False)
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
            period: _compute_quantile_df(factor_as_quantile, period_returns, quantiles=quantiles)
            for period, period_returns in forward_returns.items()
        }
    )
