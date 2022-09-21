from math import floor

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit, TimeSeriesSplit

import utils
from timeseries_dataset import NumpyTimeSeriesDataLoader

# Load Data
spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
spy = utils.get_nonempty_numeric_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

X_0 = spy.iloc[0]  # record initial raw X values

# brn = utils.generate_brownian_motion(len(spy), len(spy.columns), initial=X_0.to_numpy())
# print(len(spy))

pct_df = spy.pct_change()[1:]  # Compute percent change
pct_df = utils.remove_outliers(pct_df)

X = pct_df.to_numpy()[:-1]

y = np.sign(pct_df['close'].to_numpy())[1:]

period = 5
features = X.shape[1]
loader = NumpyTimeSeriesDataLoader(X, y, period=period, test_size=.20)

# https://stackoverflow.com/questions/48390601/explicitly-specifying-test-train-sets-in-gridsearchcv
validation_split = 0.2
# The indices which have the value -1 will be kept in train.
train_indices = np.full((floor(len(loader.X_train) * (1-validation_split)),), -1, dtype=int)
# The indices which have zero or positive values, will be kept in test
validation_indices = np.full((floor(len(loader.X_train) * validation_split),), 0, dtype=int)
ps = PredefinedSplit(np.append(train_indices, validation_indices))

param_grid = dict(splitter=['best', 'random'],
                  max_depth=[5, 10, 25, None],
                  min_samples_split=[2, 5, 10, 50],
                  min_samples_leaf=[1, 5, 10])

gscv = GridSearchCV(estimator=DecisionTreeClassifier(),
                    param_grid=param_grid,
                    scoring='f1_weighted',
                    n_jobs=-1,
                    cv=TimeSeriesSplit(n_splits=5),
                    refit=True)

NUM_TRIALS = 1

best_model_score = 0
best_model = None

for i in tqdm(range(NUM_TRIALS)):
    gscv.fit(loader.X_train, loader.y_train)
    if gscv.best_score_ > best_model_score:
        best_model_score = gscv.best_score_
        best_model = gscv.best_estimator_

print(f'Best score: {best_model_score}')

predicted_y_train = best_model.predict(loader.X_train)
print('Train report:')
print(classification_report(loader.y_train, predicted_y_train))

predicted_y_test = best_model.predict(loader.X_test)
print('Test report:')
print(classification_report(loader.y_test, predicted_y_test))
