---
title: Installation
nav_order: 3
---

# Installation

```bash
# We have not released the package to pypi yet, so you need to install from source!!!
# Install from source for loacl testing!!!
## replace $ThisRepoURL with the actual repo url
git clone $ThisRepoURL 
## install from source
pip install -e .
```

# Load Data

Download the data 
from [here](https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz)

run the command and download data put in correct path automatically.

```bash
# We have not released this repo yet, so you need download the data manually!!! See command below!!!
# Auto download data
fire download
```

If you have already downloaded the data from [here](https://github.com/fire-institute/fire/releases/download/marketdata/AStockData.tar.gz), you can run the command to check the data and put the data in the correct path

```bash
# replace path_to_data.tar.gz with the actual path
fire load path_to_data.tar.gz
```
