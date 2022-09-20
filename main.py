from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler as Scaler

import utils
from timeseries_dataset import TimeSeriesDataLoader

# Load Data
spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
spy = utils.get_nonempty_float_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

X_0 = spy.iloc[0]  # record initial raw X values

# brn = utils.generate_brownian_motion(len(spy), len(spy.columns), initial=X_0.to_numpy())
# print(len(spy))

pct_df = spy.pct_change()[1:]  # Compute percent change
pct_df = utils.remove_outliers(pct_df)

X_scaler = Scaler(feature_range=(-1, 1))  # Initialize scalers for normalization
X = pct_df[["close"]][:-1].to_numpy()

y = np.sign(pct_df['close'].to_numpy())[1:] + 1
y = y.astype(np.uint8)

period = 10
features = 1
dataloader = TimeSeriesDataLoader(X, y, period=period, test_size=.20)

svc = SVC()
svc.fit(dataloader.X_train.reshape(-1, period * features), dataloader.y_train)
predicted_y_test = svc.predict(dataloader.X_test.reshape(-1, period * features))
prediction_distribution = Counter(predicted_y_test)
print("Distribution of predictions:", prediction_distribution)
print(classification_report(dataloader.y_test, predicted_y_test))
