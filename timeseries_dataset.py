from numpy.lib import stride_tricks
from sklearn.preprocessing import MinMaxScaler as Scaler
from math import floor


class TimeSeriesDataLoader:
    def __init__(self, X, y, period=100, test_size=0.2):
        train_end = test_start = floor(len(X) * (1-test_size))

        scaler = Scaler(feature_range=(-1, 1))
        scaler.fit(X[:train_end])

        X = scaler.transform(X)

        # https://stackoverflow.com/questions/43185589/sliding-windows-from-2d-array-that-slides-along-axis-0-or-rows-to-give-a-3d-arra
        nd0 = X.shape[0] - period + 1
        samples, features = X.shape
        s0, s1 = X.strides

        self.X = stride_tricks.as_strided(X, shape=(nd0, period, features), strides=(s0, s0, s1))
        self.y = y[period-1:].reshape(-1, 1)

        self.X_train = self.X[:train_end]
        self.y_train = self.y[:train_end]
        self.X_test = self.X[test_start:]
        self.y_test = self.y[test_start:]
