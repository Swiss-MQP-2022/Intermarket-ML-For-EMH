import pandas as pd
import numpy as np
from scipy import stats


def pct_to_cumulative(data, initial):
    return (data + 1).cumprod() * initial


def get_nonempty_float_columns(data):
    return data.select_dtypes(include=['float64']).dropna(axis=1, how='all')


def remove_outliers(df: pd.DataFrame, z_thresh=3) -> pd.DataFrame:
    only_float64 = get_nonempty_float_columns(df)
    z_scores = np.abs(stats.zscore(only_float64, nan_policy='omit'))
    df = df[(z_scores < z_thresh).all(axis=1)]
    return df
