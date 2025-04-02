from firefin.evaluation.academia.fama_macbeth import FamaMacBeth
from firefin.data.fake import gen_df

factor = gen_df(253, 100, index="day", mock="rand")
return_adj =   gen_df(253, 100, index="day", mock="return")

def test_fama_macbeth_regression():
    r = FamaMacBeth.run_regression(factor, return_adj, window=252)
    print(r)
    stats = FamaMacBeth.test_statistics(r)
    print(stats)

test_fama_macbeth_regression()