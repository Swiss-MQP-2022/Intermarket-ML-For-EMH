from math import floor

import numpy as np
import pandas as pd
from numpy.lib import stride_tricks

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, Subset, DataLoader

import utils
from utils import Scaler


class TimeSeriesDataset:
    def __init__(self, X, y,
                 period=100,
                 test_size=0.2,
                 scaler: Scaler = None,
                 flatten=True,
                 name=None):
        """
        :param X: input data
        :param y: target data
        :param period: period of analysis
        :param test_size: proportion of dataset to use in the testing set
        :param scaler: scaler to use with the data. Doesn't scale data if not provided (default)
        :param flatten: whether to flatten the sliding windows generated by the dataset
        :param name: name of the dataset
        """
        self.name = name  # set the dataset name
        # set the data index
        if isinstance(y, pd.Series):
            self.index = y.index  # This may be useful for aligning time series plots

        self.scaler = scaler  # set the scaler for the dataset
        self.y = y[period - 1:]  # drop unusable values for y and set y

        # get the indices of end of the training set / start of the testing set
        train_end = test_start = floor(len(X) * (1 - test_size))
        features = X.shape[1]  # get the number of features

        # Using a scaler is optional
        if self.scaler is not None:
            self.scaler.fit(X[:train_end])  # fit the scaler to the training data
            X = self.scaler.transform(X)  # scale the data
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.to_numpy()

        # https://stackoverflow.com/questions/43185589/sliding-windows-from-2d-array-that-slides-along-axis-0-or-rows-to-give-a-3d-arra
        # Generate sliding windows
        nd0 = X.shape[0] - period + 1
        s0, s1 = X.strides
        self.X = stride_tricks.as_strided(X, shape=(nd0, period, features), strides=(s0, s0, s1))

        if flatten:  # flatten the sliding windows
            self.X = self.X.reshape(-1, period * features)

        # set training and testing subsets
        self.X_train = self.X[:train_end]
        self.y_train = self.y[:train_end]
        self.X_test = self.X[test_start:]
        self.y_test = self.y[test_start:]


class MultiAssetDataset(TimeSeriesDataset):
    def __init__(self, name, symbols, data, y, period=5, scaler=None):
        """
        :param name: name of dataset
        :param symbols: list of tuples (asset type, SYMBOL)
        :param data:
        :param y:
        :param period:
        :param scaler:
        """
        self.symbols = symbols
        self.dfs = [utils.get_df_from_symbol(asset_type, symbol, data) for asset_type, symbol in self.symbols]

        X, y = utils.align_data(utils.join_datasets(self.dfs), y)

        super().__init__(X, y, period=period, scaler=scaler, name=name)


class TorchTimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, period=10):
        """
        :param X: m by n input matrix. m=number of time series, n=total data points
        :param y: n label vector
        :param period: length of time series to train on
        """
        self.X = X
        self.y = y
        self.period = period

    def __len__(self):
        m = self.X.size(0)
        return m - self.period

    def __getitem__(self, index):
        x = self.X[index:index + self.period]
        y = self.y[index + self.period]
        return x, y


class TorchTimeSeriesDataLoader:
    def __init__(self, X, y, validation_split=0.20, test_split=0.20, batch_size=4, period=100):
        self.dataset = TorchTimeSeriesDataset(X, y, period=period)

        test_start = floor(len(self.dataset) * (1 - test_split))  # Starting index of testing set
        subset_split = validation_split * (1 - test_split)  # get correct validation split after pulling test_split

        train_indices, validation_indices = train_test_split(range(test_start), test_size=subset_split)
        test_indices = range(test_start, len(self.dataset))
        self.train_dataset = Subset(self.dataset, train_indices)
        self.validation_dataset = Subset(self.dataset, validation_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_data_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.all_data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
