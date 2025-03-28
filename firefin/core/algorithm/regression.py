# -*- coding: utf-8 -*-
# @Created : 2025/3/26 17:01
# @Author  : Liao Renjie
# @Email   : liao.renjie@techfin.ai
# @File    : least_square.py
# @Software: PyCharm

import textwrap
import typing

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["least_square", "RollingRegressor", "rolling_regression", "table_regression"]

NotProvided = object()


class RegressionResult:
    def __init__(self, sm_result: sm.regression.linear_model.RegressionResults, fit_intercept: bool, univariate: bool):
        self.sm_result = sm_result
        self.fit_intercept = fit_intercept
        self.univariate = univariate

    @property
    def alpha(self):
        """float or None"""
        if self.fit_intercept:
            return self.sm_result.params[0]
        else:
            return None

    @property
    def beta(self):
        """1D array if multivariate or float if univariate"""
        if self.univariate:
            return self.sm_result.params[-1]
        else:
            # multivariate
            if self.fit_intercept:
                return self.sm_result.params[1:]
            else:
                return self.sm_result.params

    @property
    def r2(self):
        return self.sm_result.rsquared

    @property
    def r2_adj(self):
        return self.sm_result.rsquared_adj

    @property
    def residuals(self):
        return self.sm_result.resid


class BatchRegressionResult:
    def __init__(
        self,
        beta,
        alpha=None,
        r2=None,
        r2_adj=None,
        residuals=None,
    ):
        # NOTE: public names will be displayed in __repr__
        self.alpha = alpha
        self.beta = beta
        self.r2 = r2
        self.r2_adj = r2_adj
        self.residuals = residuals

    def __repr__(self):
        content = {a: getattr(self, a) for a in dir(self) if not a.startswith("_")}
        content = ",\n".join(
            [f" {k}:\n{textwrap.indent(repr(v), prefix='    ')}" for k, v in content.items() if v is not None]
        )
        return f"{self.__class__.__name__}(\n{content}\n)"


def _regression(
    x: pd.DataFrame | pd.Series, y: pd.Series, w: pd.Series = None, fit_intercept: bool = True
) -> sm.regression.linear_model.RegressionResults:
    if fit_intercept:
        x = sm.add_constant(x)
    if w is None:
        model = sm.OLS(y, x)
    else:
        model = sm.WLS(y, x, weights=w)
    return model.fit()


def least_square(
    x: pd.Series | pd.DataFrame | list[pd.Series] | np.ndarray,
    y: pd.Series | np.ndarray,
    w: pd.Series | np.ndarray | None = None,
    fit_intercept: bool = True,
) -> RegressionResult:
    """A simple wrapper around sm.OLS or sm.WLS"""

    if isinstance(x, (tuple, list)):
        x = pd.concat(x, axis=1)

    if isinstance(x, pd.Series):
        x = x.to_frame()

    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x.ndim != 2:
            raise ValueError("x must be 1d or 2d array")

    univariate = x.shape[1] == 1

    result = _regression(x, y, w=w, fit_intercept=fit_intercept)
    return RegressionResult(result, fit_intercept=fit_intercept, univariate=univariate)


class RollingRegressor:
    def __init__(
        self,
        x,
        y,
        w=None,
        *,
        mode: typing.Literal["single", "multi"] = None,
        axis=0,
        fit_intercept: bool = True,
    ):
        """
        Notes
        -----
        We generally don't check the alignment of the inputs. It's the user's obligation to make sure the inputs are
        compatible in turns of shape and align with each other.

        """
        self._keys = {}
        self._index = {}
        self._columns = {}
        self.x = self._parse_data(x, "x")
        self.y = self._parse_data(y, "y")
        self.w = self._parse_data(w, "w", allow_none=True)

        # "multi": x is 3d array
        # "single": x is 2d array
        if isinstance(self.x, np.ndarray):
            if self.x.ndim == 2:
                self.inferred_mode = "single"
                self.x = self.x.reshape(1, *self.x.shape)
            elif self.x.ndim == 3:
                self.inferred_mode = "multi"
            else:
                raise ValueError("x must be 2d or 3d array")
        else:
            raise ValueError("parsed x should be array")

        if fit_intercept:
            self.x = np.concatenate([np.ones((1, *self.x.shape[1:])), self.x])

        if mode is not None and mode != self.inferred_mode:
            raise ValueError(f"inferred mode ({self.inferred_mode}) is not equal to the specified mode ({mode})")

        self.keys = None if not self._keys else next(iter(self._keys.values()))
        self.index = None if not self._index else next(iter(self._index.values()))

        if not self._columns:
            self.columns = None
        else:
            len_col = list(map(len, self._columns.values()))
            max_len_loc = len_col.index(max(len_col))
            self.columns = list(self._columns.values())[max_len_loc]

        self.fit_intercept = fit_intercept

    @property
    def is_univariate(self):
        if self.inferred_mode == "single":
            return True
        else:
            assert self.inferred_mode == "multi"
            return False

    def _parse_data(self, a, data_name: typing.Literal["x", "y", "w"], allow_none=False):
        if a is None:
            if allow_none:
                return
            else:
                raise ValueError(f"input {data_name} cannot be None")
        if isinstance(a, pd.Series):
            a = a.to_frame()
        if isinstance(a, pd.DataFrame):
            self._index[data_name] = a.index
            self._columns[data_name] = a.columns
            return a.values
        elif isinstance(a, np.ndarray):
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            if a.ndim not in (2, 3):
                raise ValueError(f"input {data_name} should be 2-d or 3-d if it's array")
            return a
        else:
            if data_name in ("x", "w"):
                if isinstance(a, dict):
                    self._keys[data_name] = list(a.keys())
                    a = list(a.values())

                if isinstance(a, (list, tuple)):
                    if len(set([i.shape for i in a])) != 1:
                        raise ValueError(f"input {data_name} should have same shape")
                    if not all(i.ndim == 2 for i in a):
                        raise ValueError(f"input contents of {data_name} should be 2-d, if it's list")
                    if isinstance(a[0], pd.DataFrame):
                        self._index[data_name] = a[0].index
                        self._columns[data_name] = a[0].columns
                    a = np.array(a)
                    return a
                else:
                    raise TypeError(f"input {data_name} should be array-like or list")

            else:
                raise ValueError(f"input {data_name}'s type not supported")

    @classmethod
    def _transpose_or_none(cls, _x):
            # the last 2 axes are always time x stocks
            if _x is not None:
                return np.swapaxes(_x, -1, -2)

    def fit(self, window: int | None = None, axis=0):
        x = self.x
        y = self.y
        w = self.w

        keys = self.keys
        index = self.index
        columns = self.columns

        fit_intercept = self.fit_intercept
        univariate = self.inferred_mode == "single"
        transpose = axis != 0

        if transpose:
            x = self._transpose_or_none(x)
            y = self._transpose_or_none(y)
            w = self._transpose_or_none(w)

        # generic shape compat
        k, n1, m1 = x.shape
        n2, m2 = y.shape
        if m1 != m2 and min(m1, m2) != 1:
            raise ValueError(f"incompatible x, y shapes: {x.shape} vs {y.shape}")
        if n1 != n2:
            raise ValueError(f"x, y should have same length")

        n = n1
        m = max(m1, m2)
        m3 = 1

        if w is not None:
            n3, m3 = w.shape
            if m3 > 1 and m3 != m:
                raise ValueError(f"incompatible x, y, w shapes: {x.shape} vs {y.shape} vs {w.shape}")
            if n3 != n:
                raise ValueError(f"x, w should have same length")


        # window not specified, use total length as window
        # in this case, result should also be pruned
        is_table = window is None
        if is_table:
            window = n

        alpha = None
        if fit_intercept:
            alpha = np.full((n, m), np.nan)
        beta = np.full((k - fit_intercept, n, m), np.nan)

        for i in range(n - window + 1):
            x = x[:, i : i + window]
            y = y[i : i + window]
            w = None if w is None else w[i : i + window]
            for j in range(m):
                x_j = x[:, :, min(j, m1 - 1)].T
                y_j = y[:, min(j, m2 - 1)]
                w_j = None if w is None else w[:, min(j, m3 - 1)]
                res = RegressionResult(
                    # fit_intercept is always False, because we've padded X in __init__
                    _regression(x_j, y_j, w_j, fit_intercept=False),
                    fit_intercept=fit_intercept,
                    univariate=univariate,
                )
                alpha[i + window - 1, j] = res.alpha
                beta[:, i + window - 1, j] = res.beta

        ### squeeze if table
        if is_table:
            # columns
            alpha = alpha[-1]
            # keys x columns
            beta = beta[:, -1]
        ### maybe transpose back
        if transpose:
            beta = self._transpose_or_none(beta)
        ### wrap dataframe if possible
        if is_table:
            alpha = pd.Series(alpha, index=index if transpose else columns, name="alpha")
            if transpose:
                # axis = 1
                beta = pd.DataFrame(beta, index=index, columns=keys)
            else:
                beta = pd.DataFrame(beta, index=keys, columns=columns)
            if self.is_univariate:
                beta = beta.squeeze(axis=axis)

        return BatchRegressionResult(beta, alpha=alpha)


def rolling_regression(x, y, window, w=None, *, fit_intercept=True):
    return RollingRegressor(x, y, w, fit_intercept=fit_intercept).fit(window)


def table_regression(x, y, w=None, *, fit_intercept=True, axis=0):
    return RollingRegressor(x, y, w, fit_intercept=fit_intercept).fit(None, axis=axis)
