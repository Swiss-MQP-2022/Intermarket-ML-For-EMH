import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from math import floor


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
        m = self.X.size(0)
        return m - self.period

    def __getitem__(self, index):
        x = self.X[index:index+self.period]
        y = self.y[self.period]
        return x, y


class TimeSeriesDataLoader:
    def __init__(self, X, y, validation_split=0.20, test_split=0.20, batch_size=4, period=100):
        self.dataset = TimeSeriesDataset(X, y, period=period)

        test_start = floor(len(self.dataset) * (1 - test_split))  # Starting index of testing set
        subset_split = validation_split*(1-test_split)  # math to get correct validation split after pulling test_split

        train_indices, validation_indices = train_test_split(range(test_start), test_size=subset_split)
        test_indices = range(test_start, len(self.dataset))
        self.train_dataset = Subset(self.dataset, train_indices)
        self.validation_dataset = Subset(self.dataset, validation_indices)
        self.test_dataset = Subset(self.dataset, test_indices)

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_data_loader = DataLoader(self.validation_dataset, batch_size=batch_size, shuffle=True)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
        self.all_data_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
