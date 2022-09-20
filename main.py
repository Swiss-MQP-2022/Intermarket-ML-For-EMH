import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, PredefinedSplit

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

X = pct_df.to_numpy()[:-1]

y = np.sign(pct_df['close'].to_numpy())[1:]

period = 5
features = X.shape[1]
loader = TimeSeriesDataLoader(X, y, period=period, test_size=.20)

# kfolds = 10

# https://stackoverflow.com/questions/48390601/explicitly-specifying-test-train-sets-in-gridsearchcv
# The indices which have the value -1 will be kept in train.
train_indices = np.full((len(loader.X_train),), -1, dtype=int)
# The indices which have zero or positive values, will be kept in test
test_indices = np.full((len(loader.X_test),), 0, dtype=int)
ps = PredefinedSplit(np.append(train_indices, test_indices))

gscv = GridSearchCV(estimator=DecisionTreeClassifier(),
                    param_grid=dict(
                        splitter=['best', 'random'],
                        max_depth=[5, 10, None],
                        min_samples_split=[2, 5, 10, 50],
                        min_samples_leaf=[1, 5, 10]
                    ),
                    scoring='f1_weighted',
                    n_jobs=-1,
                    cv=ps,
                    refit=True,
                    return_train_score=True)

# NOTE: Need to provide full X and y when using PredefinedSplit for GridSearchCV
gscv.fit(loader.X.reshape(-1, period * features), loader.y)

model = gscv.best_estimator_

print(f'Mean CV score of best model: {gscv.best_score_}')
print(f'Best parameters: {gscv.best_params_}')
# print('All CV results: ', gscv.cv_results_)

# plt.figure(figsize=(20, 15))
# tree.plot_tree(model, ax=plt.gca(), fontsize=10)
# plt.tight_layout()
# plt.show()

print(f'Feature importance: {model.feature_importances_}')

predicted_y_train = model.predict(loader.X_train.reshape(-1, period * features))
print('Train report:')
print(classification_report(loader.y_train, predicted_y_train))

predicted_y_test = model.predict(loader.X_test.reshape(-1, period * features))
print('Test report:')
print(classification_report(loader.y_test, predicted_y_test))
