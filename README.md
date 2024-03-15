# NaviQuant

This repo is the bundled opensource toolkit for book _Navigate through the Factor Zoo: The Science of Factor Investing_.

## Installation
```bash
pip install naviquant

# for loacl testing
pip install -e .
```

## Usage
Download the data from [here](https://github.com/auderson/FactorInvestmentResearchEngine/releases/download/marketdata/AStockData.tar.gz)

run the command and download data put in correct path automatically.
```bash
# Auto download data
fire download
```
if you have already downloaded the data, you can run the command to check the data and put the data in the correct path 
```bash
fire load path_to_data.tar.gz
```


start to code
```python
import naviquant as nq

# get data
close = nq.fetch_data('close')['close']
open = nq.fetch_data('open')['open']

# compute forward returns
fr = nq.compute_forward_returns(open, [1, 5, 10])

# evaluate factor
mng = nq.Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)

```

## Features
1. handy functions for fast factor computation
2. various tools for factor evaluation


