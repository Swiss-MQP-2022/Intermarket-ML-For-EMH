from pathlib import Path
from typing import Protocol, Union, TypeVar, Callable

import pandas as pd
import numpy as np
from scipy import stats

from sklearn.decomposition import PCA

from constants import DATASET_SYMBOLS, DataDict, AssetID


class Scaler(Protocol):
    def fit(self, X): ...
    def transform(self, X) -> np.ndarray: ...
    def fit_transform(self, X) -> np.ndarray: ...
    def inverse_transform(self, X) -> np.ndarray: ...


def pct_to_cumulative(data, initial=None):
    """
    Converts percent change data to cumulative raw data values
    :param data: data to convert
    :param initial: initial value to cumulate from
    :return: cumulative raw converted data
    """
    cumulative = (data + 1).cumprod(axis=0)
    if initial is not None:
        cumulative *= initial
    return cumulative


def generate_brownian_motion(samples, feature_count, mu=1e-5, sigma=1e-3, cumulative=True, initial=None):
    """
    Generates a brownian motion dataset
    :param samples: number of samples to generate
    :param feature_count: number of features to generate
    :param mu: mean to use when generating percent change
    :param sigma: standard deviation to use when generating percent change
    :param cumulative: whether to generate cumulative data instead of returning percent change
    :param initial: initial values to use if cumulative is True
    :return: brownian motion data
    """
    norm = np.random.normal(loc=mu, scale=sigma, size=(samples, feature_count))
    return pct_to_cumulative(norm, initial) if cumulative else norm


def remove_outliers(df: pd.DataFrame, z_thresh=3) -> pd.DataFrame:
    """
    Remove outliers from a provided dataframe
    :param df: dataframe to remove outliers from
    :param z_thresh: z-score threshold to consider outliers beyond
    :return: filtered dataframe
    """
    attrs = df.attrs
    only_numeric = get_nonempty_numeric_columns(df)  # only consider non-empty numeric columns
    z_scores = np.abs(stats.zscore(only_numeric, nan_policy='omit'))  # calculate z-scores
    df = df[(z_scores < z_thresh).all(axis=1)]  # filter to only non-outliers
    df.attrs = attrs
    return df


def get_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    filter dataframe to only numeric columns
    :param data: dataframe to filter
    :return: filtered dataframe
    """
    return data.select_dtypes(include=np.number)


def get_nonempty_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    filter dataframe to non-empty numeric columns (NOTE: code treats columns where all values are the same as "empty")
    :param data: dataframe to filter
    :return: filtered dataframe
    """
    df = get_numeric_columns(data)
    n_unique = df.nunique()

    return df.drop(n_unique[n_unique == 1].index, axis=1)


def load_data(path=r'./data', set_index_to_date=True, zero_col_thresh=1) -> DataDict:
    """
    Load all data into a 2D dictionary of data
    :param path: path to directory containing data
    :param set_index_to_date: dictionary containing desired columns to make index of dataframes. Does not set index if False, uses default_index_cols if True
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: 2D dictionary of data, where the first key is the asset type (first folder level), second key is asset name (csv name)
    """
    path = Path(path)

    data = {}
    for asset_type in path.iterdir():  # for each folder (asset type) in data directory
        data[asset_type.name] = {}
        for filename in asset_type.glob("*.csv"):  # for each file (asset) in folder
            asset_name = filename.stem  # get asset name

            df = pd.read_csv(filename, header=0)  # load to dataframe
            df.attrs['name'] = asset_name  # set the name attribute of the dataframe (used for joining datasets)

            if set_index_to_date:
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
                df = df.set_index('date')  # set index to date column

            df.drop_duplicates(inplace=True)  # drop duplicate rows
            df = drop_zero_cols(df, thresh=zero_col_thresh)

            data[asset_type.name][asset_name] = df  # save data to dictionary

    return data


def get_df_from_symbol(asset_type: str, symbol: str, data: DataDict) -> pd.DataFrame:
    """
    Helper function for quickly getting dfs from symbols
    :param asset_type: asset type
    :param symbol: EXCHANGE.SYMBOL
    :param data: data object from load_data()
    :return: specified data frame
    """
    return data[asset_type][symbol]


def make_percent_series(data: pd.Series, fill_method=None) -> pd.Series:
    """
    Converts a series to percent-change
    :param data: series to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :return:
    """
    if fill_method is None:  # no fill method provided
        data = data.dropna()  # drop NaNs
        data = data[data != 0]  # drop zeros
    else:  # fill method provided
        data = data.replace(0, method=fill_method)  # replace zeros using fill method

    data = data.pct_change(fill_method=fill_method)  # compute percent change (fills NaNs if fill_method is provided).
    return data.iloc[1:]  # Remove first value (always NaN after computing percent change)


def make_percent_data(df: pd.DataFrame, fill_method=None, zero_col_thresh=1) -> pd.DataFrame:
    """
    Convert a dataframe to percent-change
    :param df: dataframe to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: percent-change data
    """
    attrs = df.attrs

    df = get_nonempty_numeric_columns(df)  # Filter to only non-empty numeric columns
    if zero_col_thresh:  # ignore columns with lots of zeros if a threshold has been set
        df = drop_zero_cols(df, thresh=zero_col_thresh)

    if fill_method is None:  # no fill method has been set
        df = df.dropna()  # drop NaNs
        df = drop_zero_rows(df)  # drop zeros
    else:  # fill method provided
        df = df.replace(0, method=fill_method)  # replace zeros using fill method

    df = df.pct_change(fill_method=fill_method)  # compute and return percent change (uses fill method if provided)

    df.attrs = attrs
    return df.iloc[1:]  # Remove first value (always NaN after computing percent change)


T = TypeVar('T')


def map_data_dict(data: DataDict,
                  map_func: Callable[[pd.DataFrame, any], T],
                  **kwargs) -> dict[str, dict[str, T]]:
    """
    Maps a data dictionary based on the provided function
    :param data: dictionary of data to map
    :param map_func: mapping function to apply to each dataframe
    :param kwargs: keyword arguments to pass to the mapping function
    :return: mapped data dictionary
    """
    new_data = {}

    for asset_type in data.keys():  # for each asset type
        new_data[asset_type] = {}
        for asset_name in data[asset_type].keys():  # for each asset
            # apply map_func
            new_data[asset_type][asset_name] = map_func(data[asset_type][asset_name], **kwargs)

    return new_data


def make_percent_dict(data: DataDict, fill_method=None, zero_col_thresh=1) -> DataDict:
    """
    Convert a dictionary of data to percent-change
    :param data: dictionary of data to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: percent-change data dictionary
    """

    return map_data_dict(data, make_percent_data,
                         fill_method=fill_method, zero_col_thresh=zero_col_thresh)


def drop_zero_cols(df: pd.DataFrame, thresh) -> pd.DataFrame:
    """
    Removes columns containing a high proportion of zeros from a dataframe
    :param df: dataframe to remove zeros from
    :param thresh: proportion of column that must be zero to drop it
    :return: filtered dataframe
    """
    return df.drop(columns=df.columns[df[df == 0].count(axis=0) / len(df.index) > thresh])


def drop_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing zero from a dataframe
    :param df: dataframe to remove zeros from
    :return: filtered dataframe
    """
    return df[~(df == 0).any(axis=1)]


def remove_outliers_dict(data: DataDict, z_thresh=3) -> DataDict:
    """
    Remove outliers from a dictionary of data
    :param data: dictionary of data to filter
    :param z_thresh: z-score threshold to consider outliers beyond
    :return: dictionary of data with outliers removed
    """
    return map_data_dict(data, remove_outliers, z_thresh=z_thresh)


def join_datasets(data: list[pd.DataFrame], flatten_columns=True):
    """
    Join a list of datasets
    :param data: list of dataframes to join
    :param flatten_columns: whether to flatten the column names
    :return: Joined dataset
    """
    joined = pd.concat(data, axis=1, join='inner', keys=[df.attrs['name'] for df in data])
    if flatten_columns:
        joined.columns = joined.columns.to_flat_index()

    return joined


def make_pca_data(df: pd.DataFrame, target: pd.Series = None, scaler: Scaler = None, **kwargs):
    """
    Performs principal component analysis (PCA) on the provided data
    :param df: data to perform PCA on
    :param target: Series to filter index by
    :param scaler: normalization scaler to use before performing PCA
    :param kwargs: keyword arguments to pass to PCA
    :return: (dataframe of principal components, filtered target), (fitted PCA object, fitted scaler)
    """
    if target is not None:  # filter df to intersection with target if provided
        df, target = align_data(df, target)

    index = df.index  # get index before performing PCA

    if scaler is not None:  # normalize df using scaler if provided
        df = scaler.fit_transform(df)

    pca = PCA(**kwargs)  # initialize PCA instance
    principal_components = pca.fit_transform(df)  # perform PCA
    principal_df = pd.DataFrame(data=principal_components, index=index)  # convert to dataframe with original index

    return (principal_df, target), (pca, scaler)


def align_data(X: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series]) -> (Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]):
    """
    Align and filter two sets of data based on their shared indices
    :param X: first dataset to align
    :param y: second dataset to align
    :return: aligned datasets
    """
    intersection = X.index.intersection(y.index)
    return X.loc[intersection], y.loc[intersection]


def generate_symbol_list(asset_types: tuple[str, ...]) -> tuple[list[AssetID], str]:
    """
    Generate a list of symbols based on a list of desired asset types
    :param asset_types: list of asset types to get symbols for
    :return: list of symbols, name of set (used for making Datasets)
    """
    name = f'[{str.join(", ", asset_types)}]'

    symbols = []
    for asset_type in asset_types:
        symbols.extend(DATASET_SYMBOLS[asset_type])

    return symbols, name
