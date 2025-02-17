import pandas as pd

class Portfolio_Sort:
    def __init__(self, fct_df):
        self.fct_df = fct_df
    
    def ranking(self):
        ranked_df = self.fct_df.rank(axis=1, method='first', ascending=False)
        return ranked_df
  
    def grouping(self, L, dup='drop'):
        deciles_df = self.fct_df.apply(lambda x: pd.qcut(x, L, labels=False, duplicates=dup), axis=1)
        return deciles_df