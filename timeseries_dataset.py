import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler as Scaler
from math import floor

import utils


class TimeSeriesDataLoader:
    def __init__(self, X, y, period=100, validation_size=0.2, test_size=0.2, random_state=1130):
        items, features = X.shape
        trim_x = items % period
        X = X[:-trim_x]

        self.X = X.reshape(-1, period, features)
        self.y = y[:-trim_x].reshape(-1, 1)
        datset_length, period_length, features = self.X.shape

        test_start = floor(datset_length * (1 - test_size))  # Starting index of testing set
        # subset_split = validation_size * (1 - test_size)
        # train_indices, validation_indices = train_test_split(range(test_start), test_size=subset_split)
        train_indices = range(test_start)
        test_indices = range(test_start, len(self.X))

        self.X_train = self.X[train_indices]
        self.y_train = self.y[train_indices]
        # self.X_val = self.X[validation_indices]
        # self.y_val = self.y[validation_indices]
        self.X_test = self.X[test_indices]
        self.y_test = self.y[test_indices]

        scaler = Scaler(feature_range=(-1, 1))

        def scale_x(x_input):
            x_shape = x_input.shape
            x_reshaped = x_input.reshape(-1, 1)
            x_scaled = scaler.fit_transform(x_reshaped)
            x_out = x_scaled.reshape(x_shape)
            return x_out

        for feature in range(features):
            x = self.X_train[:, :, feature]
            self.X_train[:, :, feature] = scale_x(x)

            # x = self.X_val[:, :, feature]
            # self.X_val[:, :, feature] = scale_x(x)

            x = self.X_test[:, :, feature]
            self.X_test[:, :, feature] = scale_x(x)


if __name__ == "__main__":
    spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
    spy = utils.get_nonempty_float_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

    pct_df = spy.pct_change()[1:]  # Compute percent change
    pct_df = utils.remove_outliers(pct_df)

    X = pct_df[:-1].to_numpy()
    y = np.sign(pct_df['close'].to_numpy())[1:] + 1

    dataloader = TimeSeriesDataLoader(X, y)
