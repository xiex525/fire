# Fama-MacBeth Regression

## Overview
This Python script implements the **Fama-MacBeth** regression procedure, which is commonly used in asset pricing models. The method evaluates the risk premia for different factors by performing a time-series of cross-sectional regressions on asset returns. The script uses the **LinearRegression** from scikit-learn to run the regressions.

## Class: `FamaMacbeth`

The class `FamaMacbeth` is designed to take asset returns and one or more factor dataframes, and it runs the Fama-MacBeth regression to estimate factor loadings (betas) over time.

### Parameters:

- `return_df` (pandas DataFrame): 
  - A DataFrame of asset returns with time as the index (rows) and assets as columns.
  - Example: Each column represents the returns of a specific asset, and each row corresponds to the return of these assets at a specific time.

- `*dfs` (pandas DataFrames):
  - One or more DataFrames representing factor data, where each DataFrame corresponds to a different factor.
  - Each factor DataFrame should have time as the index (rows) and assets as columns.
  - The number of factor DataFrames can vary depending on the number of factors you want to use in your asset pricing model.

### Methods:

#### 1. `__init__(self, return_df, *dfs)`
The constructor initializes the FamaMacbeth class with the return data and factor data. It stores the return DataFrame and creates a dictionary for the factor DataFrames.

- **Inputs**:
  - `return_df`: Asset returns DataFrame.
  - `*dfs`: One or more factor DataFrames.

- **Outputs**:
  - Creates an object with attributes:
    - `self.return_df`: The DataFrame containing asset returns.
    - `self.factors`: A dictionary where each factor DataFrame is stored with keys as `df1`, `df2`, etc.

#### 2. `create_df_dict(self, *dfs)`
This method organizes the provided factor DataFrames into a dictionary.

- **Inputs**:
  - `*dfs`: One or more factor DataFrames.
  
- **Outputs**:
  - Returns a dictionary (`df_dict`) where each key is `df1`, `df2`, etc., corresponding to each factor DataFrame.

#### 3. `run_regression(self)`
This method runs the Fama-MacBeth regression. It performs the following steps:
- Iterates over each time period.
- For each time period (t), it runs a cross-sectional regression of asset returns on the factors.
- Returns a DataFrame of the estimated betas (factor loadings) over time.

- **Outputs**:
  - A DataFrame of `betas`, where each column represents the estimated factor loading for a particular factor across time periods.

### Example Usage:

```python
# Assuming you have a DataFrame `returns_df` for asset returns
# and factor DataFrames `factor1_df`, `factor2_df` for factors

# Initialize the FamaMacbeth object
fama_macbeth_model = FamaMacbeth(return_df=returns_df, factor1_df=factor1_df, factor2_df=factor2_df)

# Run the regression
betas = fama_macbeth_model.run_regression()

# `betas` will contain the estimated factor loadings (betas) for each factor
