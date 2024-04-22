# Licensed under the Apache License: http://www.apache.org/licenses/LICENSE-2.0
# For details: https://github.com/fire-institute/FactorInvestmentResearchEngine/blob/master/NOTICE.txt

import os

import pandas as pd


def load_data_maps() -> dict[str, str]:
    return {
        "open": "file::feather",
        "high": "file::feather",
        "low": "file::feather",
        "close": "file::feather",
        "volume": "file::feather",
        "money": "file::feather",
        "return_adj": "file::feather",
        "adj_factor": "file::feather",
    }


def load_AStock_info() -> tuple[pd.DataFrame, pd.DataFrame]:
    current_dir = os.path.dirname(__file__)
    try:
        columns = pd.read_feather(os.path.join(current_dir, "./raw/columns.feather"))
        index = pd.read_feather(os.path.join(current_dir, "./raw/index.feather"))
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}, please download data first")
    return columns, index
