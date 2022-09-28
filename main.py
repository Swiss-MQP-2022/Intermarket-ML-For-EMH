from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

import utils
from dataset import TimeSeriesDataset, MultiAssetDataset
from trainer import ScikitModelTrainer, DataSplit

dataset_symbol_list = {
    # "simple": [("stock", "SPY.US")],
    "forex": [("stock", "SPY.US"), ("forex", "USDGBP.FOREX"), ("forex", "USDEUR.FOREX")],
    "bond": [("stock", "SPY.US"), ("bond", "US10Y.GBOND"), ("bond", "US5Y.GBOND")],
    "future": [("stock", "SPY.US"), ("future", "ES.COMM"), ("future", "NK.COMM"), ("future", "RTY.COMM")]
}

# Load all the data
all_data = utils.load_data()

# Generate the FULL available y set
y_base = utils.make_percent_series(all_data['stock']['SPY.US']['close'])
y_base = y_base.apply(np.sign).shift(-1).iloc[:-1]
y_base = y_base.replace(0, -1)  # replace 0s with -1 so classification is binary

# Scaling pipeline for PCA. If used with MultiAssetDataset, applies *after* join
pca_pipeline = make_pipeline(StandardScaler(),
                             PCA(svd_solver='full', n_components=0.95))

# 5 random data features (chosen arbitrarily)
brn_features = 5

# Generate brownian motion
brn_raw_X = utils.generate_brownian_motion(len(y_base), brn_features, cumulative=True)

# Generate normal distribution sample
norm_pct_X = utils.generate_brownian_motion(len(y_base), brn_features)

# Load raw S&P 500 data
spy_raw_X, spy_raw_y = utils.align_data(all_data['stock']['SPY.US'], y_base)

# Generate percent change on S&P 500 data
spy_pct_X, spy_pct_y = utils.align_data(utils.make_percent_data(all_data['stock']['SPY.US']), y_base)

period = 5

datasets = [
    TimeSeriesDataset(brn_raw_X, y_base, period=period, scaler=StandardScaler(), name='Brownian Motion'),
    TimeSeriesDataset(brn_raw_X, y_base, period=period, scaler=deepcopy(pca_pipeline), name='Brownian Motion PCA'),
    TimeSeriesDataset(norm_pct_X, y_base, period=period, scaler=StandardScaler(), name='Normal Sample'),
    TimeSeriesDataset(norm_pct_X, y_base, period=period, scaler=deepcopy(pca_pipeline), name='Normal Sample PCA'),
    TimeSeriesDataset(spy_raw_X, spy_raw_y, period=period, scaler=StandardScaler(), name='SPY Raw'),
    TimeSeriesDataset(spy_raw_X, spy_raw_y, period=period, scaler=deepcopy(pca_pipeline), name='SPY Raw PCA'),
    TimeSeriesDataset(spy_pct_X, spy_pct_y, period=period, scaler=StandardScaler(), name='SPY %'),
    TimeSeriesDataset(spy_pct_X, spy_pct_y, period=period, scaler=deepcopy(pca_pipeline), name='SPY % PCA'),
]

# datasets = [MultiAssetDataset(key, symbols, all_data, spy_pct_y) for key, symbols in dataset_symbol_list.items()]

models = [
    dict(estimator=DecisionTreeClassifier(),
         param_grid=dict(splitter=['best', 'random'],
                         max_depth=[5, 10, 25, None],
                         min_samples_split=[2, 5, 10, 50],
                         min_samples_leaf=[1, 5, 10])),

    # dict(estimator=SVC()),
    # dict(estimator=KNN()),
    # dict(estimator=LogisticRegression())
]

reports = {}

for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}

    for data in datasets:
        print(f'Fitting {estimator_name} on {data.name}{" using GridSearchCV" if "param_grid" in model.keys() else ""}...')

        clf = trainer.train(data.X_train, data.y_train)
        predicted_y_train = clf.predict(data.X_train)
        predicted_y_test = clf.predict(data.X_test)
        reports[estimator_name][data.name] = {
            DataSplit.TRAIN: classification_report(data.y_train, predicted_y_train, zero_division=0, output_dict=False),
            DataSplit.TEST: classification_report(data.y_test, predicted_y_test, zero_division=0, output_dict=False)
        }

        print(classification_report(data.y_test, predicted_y_test, zero_division=0))

print('Done!')

for model in reports.keys():
    for data in reports[model].keys():
        print('\n')
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            print(f'{model}: {data}, {split}')
            print(reports[model][data][split])
