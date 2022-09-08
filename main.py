import pandas as pd
import numpy as np
from timeseries_dataset import TimeSeriesDataLoader

df = pd.read_csv(r"data/stock/SPY.US.csv")
closing_prices = df["close"].to_numpy()
X = np.expand_dims(closing_prices, axis=0)
y = closing_prices

dataloader = TimeSeriesDataLoader(X, y)