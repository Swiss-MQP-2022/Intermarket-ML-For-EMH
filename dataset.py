from itertools import filterfalse
from math import floor
from typing import Union

import numpy as np
from numpy.lib import stride_tricks
import pandas as pd
from more_itertools import powerset

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

import utils
from utils import Scaler
from constants import DataDict, AssetID, DATASET_SYMBOLS


class TimeSeriesDataset:
    def __init__(self,
                 X: Union[np.ndarray, pd.DataFrame, pd.Series],
                 y: Union[np.ndarray, pd.DataFrame, pd.Series],
                 period=5,
                 test_size=0.2,
                 scaler: Scaler = None,
                 fit=True,
                 flatten=True,
                 name=None,
                 clone_scaler=False):
        """
        Dataset class for time series data (note: assumes data is already aligned)
        :param X: input data
        :param y: target data
        :param period: period of analysis
        :param test_size: proportion of dataset to use in the testing set
        :param scaler: scaler to use with the data. Doesn't scale data if not provided (default)
        :param flatten: whether to flatten the sliding windows generated by the dataset
        :param name: name of the dataset
        :param clone_scaler: whether to clone the scaler provided (prevents aliasing when building datasets in loops)
        """
        self.name = name  # set the dataset name
        self.scaler = clone(scaler) if clone_scaler else scaler  # set the scaler for the dataset
        self.period = period
        self.y = y[self.period - 1:]  # drop unusable values for y and set y

        # get the indices of end of the training set / start of the testing set
        train_end = test_start = floor(len(X) * (1 - test_size))

        if scaler is not None:
            if fit:
                self.scaler.fit(X[:train_end])  # fit the scaler to the training data
            X = self.scaler.transform(X)  # scale the data

        if isinstance(X, (pd.DataFrame, pd.Series)):  # Convert to numpy if DataFrame or Series
            X = X.to_numpy()

        if X.ndim == 1:  # if passed a 1D array for X (occurs if X is a series)
            X = np.expand_dims(X, axis=1)  # add an extra dimension (required for sliding window)

        self.features = X.shape[1]  # get the number of features

        # https://stackoverflow.com/questions/43185589/sliding-windows-from-2d-array-that-slides-along-axis-0-or-rows-to-give-a-3d-arra
        # Generate sliding windows
        nd0 = X.shape[0] - self.period + 1
        s0, s1 = X.strides
        self.X = stride_tricks.as_strided(X, shape=(nd0, self.period, self.features), strides=(s0, s0, s1))

        if flatten:  # flatten the sliding windows
            self.X = self.X.reshape(-1, self.period * self.features)

        # set training and testing subsets
        self.X_train = self.X[:train_end]
        self.y_train = self.y[:train_end]
        self.X_test = self.X[test_start:]
        self.y_test = self.y[test_start:]


class MultiAssetDataset(TimeSeriesDataset):
    def __init__(self, symbols: list[AssetID], data: DataDict, y, **kwargs):
        """
        Dataset class for time series datasets with multiple assets
        :param symbols: list of tuples (asset type, SYMBOL)
        :param data: dictionary of data to pull data from based on symbols
        :param y: target data
        :param kwargs: arbitrary keyword arguments to pass to TimeSeriesDataset
        """
        self.symbols = symbols
        self.dfs = [data[asset_type][symbol] for asset_type, symbol in self.symbols]

        joined = utils.join_datasets(self.dfs)
        X, y = utils.align_data(joined, y)

        super().__init__(X, y, **kwargs)


def build_datasets(period=5,
                   rand_features=5,
                   test_size=0.2,
                   zero_col_thresh=1,
                   replace_zero=None) -> list[TimeSeriesDataset]:
    """
    Builds the full suite of datasets for experimentation
    :param period: period for sliding windows
    :param rand_features: number of features to generate for random sample datasets
    :param test_size: proportion of dataset to use in the testing set
    :param zero_col_thresh: proportion of a column that must be zero to drop it (passed to make_percent_dict)
    :param replace_zero: value to replace zeros in y_base with. No replacement if None (default)
    :return: list of datasets
    """
    # Load all the data
    raw_data = utils.load_data()
    returns_data = utils.make_returns_datadict(raw_data, zero_col_thresh=zero_col_thresh)

    # Generate the FULL available y set
    y_base = utils.make_returns_series(raw_data['stock']['SPY.US']['close'])
    y_base = y_base.apply(np.sign).shift(-1).iloc[:-1]
    if replace_zero is not None:  # replace zeros if desired
        y_base = y_base.replace(0, replace_zero)  # replace 0s with specified value

    # Initialize list of datasets with just the random normal sample data and SPY alone
    datasets = [
        TimeSeriesDataset(np.random.normal(len(y_base), size=(len(y_base), rand_features)),
                          y_base,
                          period=period,
                          test_size=test_size,
                          scaler=StandardScaler(),
                          name='Normal Sample'),
        TimeSeriesDataset(*utils.align_data(returns_data['stock']['SPY.US'], y_base),
                          period=period,
                          test_size=test_size,
                          scaler=StandardScaler(),
                          name='SPY Only')
    ]

    # MULTI-ASSET DATASET GENERATION
    # Generate powerset of available asset types (EXCLUDES SPY!!)
    asset_powerset = filterfalse(lambda x: x == (), powerset(DATASET_SYMBOLS.keys()))

    for asset_set in asset_powerset:  # for each set of assets in the powerset of asset types
        symbol_list = utils.generate_symbol_list(asset_set)  # get list of symbols for those assets

        # generate and append new MultiAssetDataset to dataset list
        datasets.append(MultiAssetDataset([('stock', 'SPY.US')] + symbol_list[0],
                                          returns_data,
                                          y_base,
                                          name=symbol_list[1],
                                          period=period,
                                          test_size=test_size,
                                          scaler=StandardScaler(),
                                          clone_scaler=True))

    return datasets
