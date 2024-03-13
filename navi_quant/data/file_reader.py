import pandas as pd
import os

data_path = os.path.join(os.path.dirname(__file__), 'raw')

def read_feather(names):
    try:
        result = {n : pd.read_feather(f"{data_path}/{n}.feather") for n in names}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}, please download data first")

    return result


def file_reader(info: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    # TODO: support other file types
    feather_reader_names = info['feather']
    return read_feather(feather_reader_names)