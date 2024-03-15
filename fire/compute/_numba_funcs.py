# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import numpy as np
from numba import njit


@njit
def factor_to_quantile(factor: np.ndarray, quantiles: int) -> np.ndarray:
    pass
