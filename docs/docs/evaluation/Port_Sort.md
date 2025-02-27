# Portfolio_Sort Class

## Overview
The `Portfolio_Sort` class is designed for handling and manipulating factor data, allowing you to rank assets and group them into quantiles (e.g., deciles) based on a given factor. This can be useful for portfolio sorting, factor analysis, and quantitative research where you wish to create rankings and groupings of assets according to their factor scores.

## Class Definition

### `__init__(self, fct_df)`
The constructor method initializes the `Portfolio_Sort` object with the factor data. 

#### Parameters:
- `fct_df` (DataFrame): A pandas DataFrame containing the factor values for different assets over time. The rows correspond to different timestamps, and the columns represent different assets. The values in the DataFrame represent the factor scores for the assets at the given time.


### `ranking(self)`
This method ranks the assets at each time point based on their factor values. The assets with higher factor values are ranked higher (rank 1 being the highest).

#### Output:
- `ranked_df` (DataFrame): A pandas DataFrame with the same shape as `fct_df`, but with ranks instead of factor values. The ranks are calculated for each time point (row), with ties broken by the 'first' method.


### `grouping(self, L, dup='drop')`
This method groups the assets into `L` quantiles based on their factor values at each time point. The default behavior is to split the assets into `L` groups (e.g., deciles if `L=10`) based on the factor scores. The `dup` parameter handles how to deal with duplicate bin edges.

#### Parameters:
- `L` (int): The number of quantiles (e.g., `L=10` for deciles). This defines how many groups the assets should be sorted into.
- `dup` (str, optional): This defines the behavior when duplicate bin edges are encountered. The default is `'drop'`, which removes duplicate labels. If set to `'raise'`, an error will be raised when duplicates are encountered.

#### Output:
- `deciles_df` (DataFrame): A pandas DataFrame with the same shape as `fct_df`, but where each value represents the quantile group for that asset at a given time point.

## Usage Example

```python
import pandas as pd
from Portfolio_Sort import Portfolio_Sort

# Example factor data
factor_data = pd.DataFrame({
    '2021-01-01': [0.5, 0.3, 0.7],
    '2021-01-02': [0.6, 0.2, 0.8],
    '2021-01-03': [0.7, 0.1, 0.9]
}, index=['Asset1', 'Asset2', 'Asset3'])

# Instantiate the Portfolio_Sort class
portfolio = Portfolio_Sort(factor_data)

# Rank assets by their factor values
ranked_df = portfolio.ranking()

# Group assets into deciles
deciles_df = portfolio.grouping(10)

print(ranked_df)
print(deciles_df)
```

---
