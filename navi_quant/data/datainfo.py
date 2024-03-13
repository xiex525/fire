from typing import List, Tuple
import pandas as pd
import os

def load_data_maps() -> List[str]:
    return {
        'open':"file::feather",
        'high':"file::feather",
        'low':"file::feather",
        'close':"file::feather",
        'volume':"file::feather",
        'money':"file::feather",
        'return_adj':"file::feather",
        'adj_factor':"file::feather",
    }

def load_AStock_info() -> Tuple[pd.DataFrame, pd.DataFrame]:
    current_dir = os.path.dirname(__file__)
    try:
        columns = pd.read_feather(os.path.join(current_dir, './raw/columns.feather'))
        index = pd.read_feather(os.path.join(current_dir, './raw/index.feather'))
    except FileNotFoundError as e: 
        raise FileNotFoundError(f"File not found: {e}, please download data first")
    return columns, index