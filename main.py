import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import utils
from timeseries_dataset import TimeSeriesDataset
from trainer import ScikitModelTrainer, DataSplit

# Load all the data
all_data = utils.load_data()

# Generate the FULL available y set
y_base = utils.make_pct_series(all_data['stock']['SPY.US']['close']).shift(-1)[:-1].apply(np.sign)

# 5 brownian motion features chosen arbitrarily
brn_features = 5

# Generate brownian motion
brn_raw_X = utils.generate_brownian_motion(len(y_base), brn_features, cumulative=True)
brn_raw_y = y_base

# Generate normal distribution sample
norm_pct_X = utils.generate_brownian_motion(len(y_base), brn_features)
norm_pct_y = y_base

# Load raw S&P 500 data
spy_raw_X = all_data['stock']['SPY.US'][:-1]
spy_raw_y = y_base.loc[spy_raw_X.index]

# Generate percent change on S&P 500 data
spy_pct_data = utils.make_pct_data(all_data['stock']['SPY.US'])[1:-1]
spy_pct_X = spy_pct_data
spy_pct_y = y_base.loc[spy_pct_data.index]

period = 5

datasets = [
    TimeSeriesDataset(brn_raw_X, brn_raw_y, period=period, name='Brownian Motion'),
    TimeSeriesDataset(norm_pct_X, norm_pct_y, period=period, name='Normal Sample'),
    TimeSeriesDataset(spy_raw_X, spy_raw_y, period=period, name='SPY Raw'),
    TimeSeriesDataset(spy_pct_X, spy_pct_y, period=period, name='SPY %')
]

models = [
    dict(estimator=DecisionTreeClassifier(),
         param_grid=dict(splitter=['best', 'random'],
                         max_depth=[5, 10, 25, None],
                         min_samples_split=[2, 5, 10, 50],
                         min_samples_leaf=[1, 5, 10])),

    dict(estimator=SVC()),
    dict(estimator=KNN()),
    dict(estimator=LogisticRegression())
]

reports = {}

for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}

    for data in datasets:
        print(
            f'Fitting {estimator_name} on {data.name}{" using GridSearchCV" if "param_grid" in model.keys() else ""}...')

        clf = trainer.train(data.X_train, data.y_train)
        predicted_y_train = clf.predict(data.X_train)
        predicted_y_test = clf.predict(data.X_test)
        reports[estimator_name][data.name] = {
            DataSplit.TRAIN: classification_report(data.y_train, predicted_y_train, zero_division=0, output_dict=True),
            DataSplit.TEST: classification_report(data.y_test, predicted_y_test, zero_division=0, output_dict=True)
        }

print('Done!')
