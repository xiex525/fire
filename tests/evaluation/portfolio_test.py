from firefin.evaluation.academia.portfolio_sort import PortfolioSort
from firefin.data.fake import gen_df



factor = gen_df(10, 100, index="day", mock="rand")
# 1, 2, 3, 4, 5 periods
forward_returns = { i: gen_df(10, 100, index="day", mock="return") for i in range(1, 6) }
market_cap = gen_df(10, 100, index="day", mock="volume")

single_sort_r = PortfolioSort.single_sort(factor, forward_returns, market_cap, quantiles=5)
statistical_r = PortfolioSort.get_statistics(single_sort_r, quantiles=5)

print(single_sort_r[1])
print(statistical_r)