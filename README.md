# F.I.R.E.

This repo is the bundled opensource toolkit for book _Navigate through the Factor Zoo: The Science of Factor Investing_.

## Installation

```bash
pip install fire

# for loacl testing
pip install -e .
```

## Usage

Download the data from [here](https://pan.baidu.com/s/1eS3Z3Y8) with password `3z3y`.(瞎写的，暂时没有)

run the command and download data put in correct path automatically.

```bash
navi download
```

start to code

```python
import fire

# get data
data = fire.fetch_data(["open", "close", "volume"])
open_price = data["open"]


def pv_corr(close, volume):
    # price volume correlation
    return close.rolling(20).corr(volume)

factor = pv_corr(data["close"], data["volume"])

# compute forward returns
fr = fire.compute_forward_returns(open_price.shift(-1), [1, 5, 10])

# evaluate factor
mng = fire.Evaluator(factor, fr)
mng.get_ic("pearson")
mng.get_quantile_returns(5)

```

## Features

1. handy functions for fast factor computation
2. various tools for factor evaluation


