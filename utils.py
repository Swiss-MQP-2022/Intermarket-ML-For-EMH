from copy import copy
from optparse import Option, OptionValueError
from pathlib import Path
from typing import Union, TypeVar, Protocol
import re

import pandas as pd
import numpy as np
from scipy import stats

from constants import DATASET_SYMBOLS, DataDict, AssetID, Model, DataSplit, METRICS


T = TypeVar('T')


class Scaler(Protocol):
    def fit(self, X): ...

    def transform(self, X) -> np.ndarray: ...

    def fit_transform(self, X) -> np.ndarray: ...

    def inverse_transform(self, X) -> np.ndarray: ...


class Estimator(Protocol):
    def fit(self, X, y): ...

    def predict(self, X) -> ...: ...

    def predict_proba(self, x) -> ...: ...


def check_model_name(_, opt, value):
    try:
        return Model(value)
    except ValueError:
        raise OptionValueError(f'option {opt}: invalid model name: {value}')


class OptionWithModel(Option):
    TYPES = Option.TYPES + ("model_name",)
    TYPE_CHECKER = copy(Option.TYPE_CHECKER)
    TYPE_CHECKER["model_name"] = check_model_name


def get_nonempty_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    filter dataframe to non-empty numeric columns (NOTE: code treats columns where all values are the same as "empty")
    :param data: dataframe to filter
    :return: filtered dataframe
    """
    df = data.select_dtypes(include=np.number)  # filter to numeric columns
    n_unique = df.nunique()  # get number of unique values per column

    return df.drop(n_unique[n_unique == 1].index, axis=1)  # remove columns with only one unique value (aka empty)


def load_data(path=r'./data') -> DataDict:
    """
    Load all data into a 2D dictionary of data
    :param path: path to directory containing data
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: 2D dictionary of data, where the first key is the asset type (first folder level), second key is asset name (csv name)
    """
    path = Path(path)

    data = {}
    for asset_type in path.iterdir():  # for each folder (asset type) in data directory
        data[asset_type.name] = {}
        for filename in asset_type.glob('*.csv'):  # for each file (asset) in folder
            asset_name = filename.stem  # get asset name

            df = pd.read_csv(filename, header=0)  # load to dataframe
            df.attrs['name'] = asset_name  # set the name attribute of the dataframe (used for joining datasets)

            df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')  # date format parsing
            df = df.set_index('date')  # set index to date column

            df = df.drop_duplicates()  # drop duplicate rows
            df = get_nonempty_numeric_columns(df)  # filter to only non-empty numeric columns

            data[asset_type.name][asset_name] = df  # store data to dictionary

    return data


def drop_zero_cols(df: pd.DataFrame, thresh) -> pd.DataFrame:
    """
    Removes columns containing a high proportion of zeros from a dataframe
    :param df: dataframe to remove zeros from
    :param thresh: proportion of column that must be zero to drop it
    :return: filtered dataframe
    """
    return df.drop(columns=df.columns[df[df == 0].count(axis=0) / len(df.index) > thresh])


def make_returns_series(data: pd.Series, fill_method=None) -> pd.Series:
    """
    Converts a series to returns
    :param data: series to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :return: returns data
    """
    if fill_method is None:  # no fill method provided
        data = data.dropna()  # drop NaNs
        data = data[data != 0]  # drop zeros
    else:  # fill method provided
        data = data.replace(0, method=fill_method)  # replace zeros using fill method

    data = data.pct_change(fill_method=fill_method)  # compute returns (fills NaNs if fill_method is provided).
    return data.iloc[1:]  # Remove first value (always NaN after computing returns)


def make_returns_dataframe(df: pd.DataFrame, fill_method=None, zero_col_thresh=1) -> pd.DataFrame:
    """
    Convert a dataframe to returns
    :param df: dataframe to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: return data
    """
    attrs = df.attrs

    if zero_col_thresh:  # ignore columns with lots of zeros if a threshold has been set
        df = drop_zero_cols(df, thresh=zero_col_thresh)  # remove columns that have too many zeros

    if fill_method is None:  # no fill method has been set
        df = df.dropna()  # drop NaNs
        df = drop_zero_rows(df)  # drop zeros
    else:  # fill method provided
        df = df.replace(0, method=fill_method)  # replace zeros using fill method

    df = df.pct_change(fill_method=fill_method)  # compute returns (uses fill method if provided)

    df.attrs = attrs
    return df.iloc[1:]  # Remove first value (always NaN after computing returns)


def make_returns_datadict(data: DataDict, fill_method=None, zero_col_thresh=1) -> DataDict:
    """
    Convert a dictionary of data to returns
    :param data: dictionary of data to convert
    :param fill_method: fill method to pass to pct_change. Drop NaNs instead if not provided (default)
    :param zero_col_thresh: proportion of a column that must be zero to drop it
    :return: returns data dictionary
    """
    new_data = {}

    for asset_type in data.keys():  # for each asset type
        new_data[asset_type] = {}
        for asset_name in data[asset_type].keys():  # for each asset
            # compute returns
            new_data[asset_type][asset_name] = make_returns_dataframe(data[asset_type][asset_name],
                                                                      fill_method=fill_method,
                                                                      zero_col_thresh=zero_col_thresh)

    return new_data


def drop_zero_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing ANY zeros from a dataframe
    :param df: dataframe to remove zeros from
    :return: filtered dataframe
    """
    return df[~(df == 0).any(axis=1)]


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


def align_data(X: Union[pd.DataFrame, pd.Series], y: Union[pd.DataFrame, pd.Series]) -> (
Union[pd.DataFrame, pd.Series], Union[pd.DataFrame, pd.Series]):
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


def make_filename_safe(name: str) -> str:
    """
    Make a string filename-safe
    :param name: string to make safe
    :return: safe filename
    """
    return re.sub('[,:]', '', name.rstrip()).replace(' ', '_')


def compute_consensus(data: pd.Series, period: int) -> pd.Series:
    """
    Compute the moving consensus (mode) of a series
    :param data: series to compute consensuses on
    :param period: period of moving window
    :return: moving consensus
    """
    windowed = data.rolling(period)
    consensus = windowed.apply(lambda x: stats.mode(x, keepdims=False)[0])
    return consensus.iloc[period - 1:]


def encode_dataset(dataset_name: str) -> list[bool]:
    """
    One-hot encodes the provided dataset name
    :param dataset_name: name of dataset to encode
    :return: dataset encoding in order defined by DATASET_SYMBOLS constant + [SPY, random]
    """
    asset_types = re.sub('[\\[\\]]', '', dataset_name).split(', ')

    encoding = [asset_type in asset_types for asset_type in DATASET_SYMBOLS.keys()]
    encoding += [False, True] if asset_types[0] == 'Normal Sample' else [True, False]

    return encoding


def encode_model(model: Model) -> list[bool]:
    """
    One-hot encodes the provided model name
    :param model: name of model to encode
    :return: model encoding in order defined by Model enum
    """
    return [ref_model == model for ref_model in Model]


def make_row_from_report(reports: dict, model: Model, dataset: str, split: DataSplit):
    """
    Generates a properly encoded data row from the provided report
    :param reports: the un-encoded report
    :param model: Model used in row
    :param dataset: Dataset used in row
    :param split: DataSplit used in row
    :return: Properly encoded result data
    """
    data = reports[model][dataset][split]

    row = encode_model(model) + encode_dataset(dataset)
    row += [split == DataSplit.TEST]
    row += [data[metric[0]][metric[1]]
            if isinstance(metric, tuple)
            else data[metric]
            for metric in METRICS.values()]

    return row


def encode_results(reports) -> pd.DataFrame:
    """
    Encodes the provided report's data into a more useful format
    :param reports: reports dictionary to encode
    :return: properly encoded results data
    """
    print('Encoding results...')

    # Get column names
    columns = [model for model in Model] + \
              list(DATASET_SYMBOLS.keys()) + \
              ['SPY', 'Random', DataSplit.TEST] + \
              list(METRICS.keys())

    # Properly encode results into useful format
    results = [make_row_from_report(reports, model, dataset, split)
               for model in reports.keys()
               for dataset in reports[model].keys()
               for split in [DataSplit.TRAIN, DataSplit.TEST]]

    return pd.DataFrame(results, columns=columns)


def save_results(data: pd.DataFrame, model: Model = None, out_dir=r'./out', prefix=None):
    """
    Generate CSVs of desired metrics
    :param data: dataframe containing metric data
    :param model: associated model (prefix for filename)
    :param out_dir: directory to save CSVs
    :param prefix: prefix to add to filename before "results" extension (used for denoting replications)
    """
    print('Saving results...')

    model_name = f'{model.value}_' if model is not None else ''
    prefix = f'{prefix}_' if prefix is not None else ''

    # Save metrics to csv
    data.to_csv(rf'{out_dir}/{make_filename_safe(model_name)}{prefix}results.csv')
