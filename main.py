import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from sklearn.preprocessing import MinMaxScaler as Scaler

import utils
import models
from timeseries_dataset import TimeSeriesDataLoader
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load Data
spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
spy = utils.get_nonempty_float_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

X_0 = spy.iloc[0]  # record initial raw X values

# brn = utils.generate_brownian_motion(len(spy), len(spy.columns), initial=X_0.to_numpy())
# print(len(spy))

pct_df = spy.pct_change()[1:]  # Compute percent change
pct_df = utils.remove_outliers(pct_df)

# brn = utils.generate_brownian_motion(len(pct_df), len(pct_df.columns), cumulative=False)

X_scaler = Scaler(feature_range=(-1, 1))  # Initialize scalers for normalization
X = pct_df[:-1].to_numpy()
# X = X_scaler.fit_transform(pct_df[:-1])  # normalize X data
y = np.sign(pct_df['close'].to_numpy())[1:] + 1

period = 25
features = 5
dataloader = TimeSeriesDataLoader(X, y, period=period)

svc = SVC()
svc.fit(dataloader.X_train.reshape(-1, period*features), dataloader.y_train)
predicted_y_test = svc.predict(dataloader.X_test.reshape(-1, period*features))
print(classification_report(dataloader.y_test, predicted_y_test))