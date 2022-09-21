from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
# from sklearn.metrics import classification_report


# def forecast_classification_report(pred, truth):
#     pred_direction = np.sign(pred)
#     truth_direction = np.sign(truth)
#
#     return classification_report(pred_direction, truth_direction)

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
    only_numeric = get_nonempty_numeric_columns(df)  # only consider non-empty numeric columns
    z_scores = np.abs(stats.zscore(only_numeric, nan_policy='omit'))  # calculate z-scores
    df = df[(z_scores < z_thresh).all(axis=1)]  # filter to only non-outliers
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


def load_data(path=r'./data', set_index_to_date=True, zero_col_thresh=1) -> dict[str, dict[str, pd.DataFrame]]:
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


def make_pct_data(df: pd.DataFrame, zero_col_thresh=1) -> pd.DataFrame:
    """
    Convert a dataframe to percent-change
    :param df: dataframe to convert
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: percent-change data
    """
    df = get_nonempty_numeric_columns(df)  # Filter to only non-empty numeric columns
    if zero_col_thresh:  # ignore columns with lots of zeros if a threshold has been set
        df = drop_zero_cols(df, thresh=zero_col_thresh)
    df = drop_zero_rows(df)  # ignore rows with zeros
    return df.dropna().pct_change()  # compute and return percent change


def make_pct_data_dict(data: dict[str, dict[str, pd.DataFrame]], zero_col_thresh=1) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Convert a dictionary of data to percent-change
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :param data: dictionary of data to convert
    :return: percent-change data dictionary
    """
    pct_data = {}

    for asset_type in data.keys():  # for each asset type
        pct_data[asset_type] = {}
        for asset_name in data[asset_type].keys():  # for each asset
            # compute and save percent-change data
            pct_data[asset_type][asset_name] = make_pct_data(data[asset_type][asset_name],
                                                             zero_col_thresh=zero_col_thresh)

    return pct_data


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


def remove_outliers_dict(data: dict[str, dict[str, pd.DataFrame]]) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Remove outliers from a dictionary of data
    :param data: dictionary of data to filter
    :return: dictionary of data with outliers removed
    """
    new_data = {}

    for asset_type in data.keys():  # for each asset type
        new_data[asset_type] = {}
        for asset_name in data[asset_type].keys():  # for each asset of the given type
            # save data with outliers removed
            new_data[asset_type][asset_name] = remove_outliers(data[asset_type][asset_name])

    return new_data


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
