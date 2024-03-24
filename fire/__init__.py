# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

from .compute.window import *
from .data.gateway import fetch_data
from .evaluation.eva_utils import compute_forward_returns, compute_ic, compute_quantile_returns
from .evaluation.evaluator import Evaluator
from .evaluation.plots import plt_ic, plt_quantile_cumulated_end_returns, plt_quantile_cumulative_returns
