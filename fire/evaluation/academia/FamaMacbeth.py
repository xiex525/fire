import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

class FamaMacbeth:
    def __init__(self, return_df, *dfs):
        self.return_df = return_df
        self.factors = self.create_df_dict(*dfs)

    def create_df_dict(self, *dfs):
        df_dict = {}
        for idx, df in enumerate(dfs):
            df_dict[f'df{idx+1}'] = df
        return df_dict
    
    def run_regression(self):
        betas = pd.DataFrame(index=self.return_df.index, columns=self.factors.keys())
        regressor = LinearRegression(fit_intercept=True)
        for t in range(len(self.return_df)):
            y = self.return_df.iloc[t, :].values
            X = pd.DataFrame()
            for factor_name, factor_df in self.factors.items():
                X[factor_name] = factor_df.iloc[t, :]
            X = X.values

            if np.any(np.isnan(X)) or np.any(np.isinf(X)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
                continue

            regressor.fit(X, y)
            for i in range(len(self.factors)):
                betas.loc[t, list(self.factors.keys())[i]] = regressor.coef_[i]
        mean_betas = betas.mean(axis=0)  
        std_betas = betas.std(axis=0) / (len(betas) ** 0.5)  

        return mean_betas, std_betas
