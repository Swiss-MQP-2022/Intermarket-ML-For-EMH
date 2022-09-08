import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split


class TimeSeriesDataset(Dataset):
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
        m, _ = self.X.shape
        return m // self.period  # number of unique periods that can be made from input data

    def __getitem__(self, index):
        start = self.period * index
        end = self.period * (index + 1)
        x = self.X[start:end, :]
        y = self.y[start:end]
        return x, y


class TimeSeriesDataLoader:
    def __init__(self, X, y, test_split=0.20, batch_size=4):
        self.dataset = TimeSeriesDataset(X, y, period=100)

        train_indices, test_indices = train_test_split(range(len(self.dataset)), test_size=test_split)
        self.train_dataset = Subset(self.dataset, train_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
