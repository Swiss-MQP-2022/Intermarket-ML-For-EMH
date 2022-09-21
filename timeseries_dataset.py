from math import floor
from typing import Protocol

import numpy as np
from numpy.lib import stride_tricks

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, Subset, DataLoader


class Scaler(Protocol):
    def fit(self, X): ...
    def transform(self, X) -> np.ndarray: ...
    def fit_transform(self, X) -> np.ndarray: ...


class NumpyTimeSeriesDataLoader:
    def __init__(self, X, y,
                 period=100,
                 test_size=0.2,
                 scaler: Scaler = MinMaxScaler(feature_range=(-1, 1)),
                 flatten=True):
        self.scaler = scaler

        train_end = test_start = floor(len(X) * (1 - test_size))
        features = X.shape[1]

        self.scaler.fit(X[:train_end])

        X = self.scaler.transform(X)

        # https://stackoverflow.com/questions/43185589/sliding-windows-from-2d-array-that-slides-along-axis-0-or-rows-to-give-a-3d-arra
        nd0 = X.shape[0] - period + 1
        s0, s1 = X.strides
        self.X = stride_tricks.as_strided(X, shape=(nd0, period, features), strides=(s0, s0, s1))

        if flatten:
            self.X = self.X.reshape(-1, period * features)

        self.y = y[period - 1:].reshape(-1, 1)

        self.X_train = self.X[:train_end]
        self.y_train = self.y[:train_end]
        self.X_test = self.X[test_start:]
        self.y_test = self.y[test_start:]


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
