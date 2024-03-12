# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/auderson/NaviQuant/blob/master/NOTICE.txt

import navi_quant as nq
import rd


factor = rd.gen_df(100, 20)
price = rd.gen_df(100, 20, mock='price')

fr = nq.compute_forward_returns(price, [1, 5, 10])

mng = nq.Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)
