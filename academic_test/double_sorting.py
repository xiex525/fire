import pandas as pd
import numpy as np

class DoubleSorting:
    def __init__(self, factor1_df, factor2_df, L1, L2):
        self.factor1_df = factor1_df
        self.factor2_df = factor2_df
        self.L1 = L1
        self.L2 = L2
    
    def rank_factors(self):
        ranked_factor1 = self.factor1_df.rank(axis=1, method='first', ascending=False)
        ranked_factor2 = self.factor2_df.rank(axis=1, method='first', ascending=False)
        return ranked_factor1, ranked_factor2
    
    def double_sort(self):
        ranked_factor1, ranked_factor2 = self.rank_factors()
        groups_factor1 = ranked_factor1.apply(lambda x: pd.qcut(x, self.L1, labels=False, duplicates='drop'), axis=1)
        sorted_groups = pd.DataFrame(index=self.factor1_df.index, columns=self.factor1_df.columns)
        for date in self.factor1_df.index:
            for group in range(self.L1):
                assets_in_group = groups_factor1.loc[date] == group
                assets_in_group_sorted = ranked_factor2.loc[date, assets_in_group].sort_values(ascending=False)
                sorted_groups.loc[date, assets_in_group_sorted.index] = pd.qcut(assets_in_group_sorted, self.L2, labels=False, duplicates='drop')
        return sorted_groups

    def calculate_group_statistics(self, sorted_groups, returns_df):
        group_returns = pd.DataFrame(index=returns_df.index, columns=range(self.L1 * self.L2))
        group_counts = pd.DataFrame(index=returns_df.index, columns=range(self.L1 * self.L2))
        for date in returns_df.index:
            for group in range(self.L1):
                for sub_group in range(self.L2):
                    assets_in_group = sorted_groups.loc[date] == (group * self.L2 + sub_group)
                    group_returns.loc[date, group * self.L2 + sub_group] = returns_df.loc[date, assets_in_group].mean()
                    group_counts.loc[date, group * self.L2 + sub_group] = assets_in_group.sum()
        return group_returns, group_counts