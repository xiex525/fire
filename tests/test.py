# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import fire


data = fire.fetch_data("close", "volume", "open")


def pv_corr(close, volume):
    return fire.ts_corr(close, volume, 5, method="pearson")


factor = pv_corr(data["close"], data["volume"])


fr = fire.compute_forward_returns(data["open"].shift(-1), [1, 5, 10])

mng = fire.Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)
