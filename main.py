from copy import deepcopy

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from copy import deepcopy

import utils
from dataset import TimeSeriesDataset, MultiAssetDataset
from trainer import ScikitModelTrainer, DataSplit
from classification_graphs import graph_classification_reports

dataset_symbol_list = {
    # "simple": [("stock", "SPY.US")],
    "forex": [("stock", "SPY.US"), ("forex", "USDGBP.FOREX"), ("forex", "USDEUR.FOREX")],
    "bond": [("stock", "SPY.US"), ("bond", "US10Y.GBOND"), ("bond", "US5Y.GBOND")],
    "future": [("stock", "SPY.US"), ("future", "ES.COMM"), ("future", "NK.COMM"), ("future", "RTY.COMM")]
}

# Load all the data
all_data = utils.load_data()

# Generate the FULL available y set
y_base = utils.make_percent_series(all_data['stock']['SPY.US']['close']).shift(-1).apply(np.sign)[:-1]

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
    TimeSeriesDataset(norm_pct_X, y_base, period=period, scaler=StandardScaler(), name='Normal Sample'),
    TimeSeriesDataset(spy_raw_X, spy_raw_y, period=period, scaler=StandardScaler(), name='SPY Raw'),
    TimeSeriesDataset(spy_raw_X, spy_raw_y, period=period, scaler=deepcopy(pca_pipeline), name='SPY Raw PCA'),
    TimeSeriesDataset(spy_pct_X, spy_pct_y, period=period, scaler=StandardScaler(), name='SPY %'),
    TimeSeriesDataset(spy_pct_X, spy_pct_y, period=period, scaler=deepcopy(pca_pipeline), name='SPY % PCA'),
    TimeSeriesDataset(brn_raw_X, y_base, period=period, scaler=deepcopy(pca_pipeline), name='Brownian Motion PCA'),
    TimeSeriesDataset(norm_pct_X, y_base, period=period, scaler=deepcopy(pca_pipeline), name='Normal Sample PCA')
]

# datasets = [MultiAssetDataset(key, symbols, all_data, spy_pct_y) for key, symbols in dataset_symbol_list.items()]

models = [
    dict(estimator=DecisionTreeClassifier(),
         param_grid=dict(splitter=['best', 'random'],
                         max_depth=[5, 10, 25, None],
                         min_samples_split=[2, 5, 10, 50],
                         min_samples_leaf=[1, 5, 10])),

    dict(estimator=SVC(probability=True)),
    dict(estimator=KNN()),
    dict(estimator=LogisticRegression(max_iter=1000))
]

reports = {}
roc_data = []
for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}
    roc_data.append([])

    for data in datasets:
        print(f'Fitting {estimator_name} on {data.name}{" using GridSearchCV" if "param_grid" in model.keys() else ""}...')
        clf = trainer.train(data.X_train, data.y_train)
        predicted_y_train = clf.predict(data.X_train)
        predicted_y_test = clf.predict(data.X_test)
        y_score = clf.predict_proba(data.X_test)
        roc_data[-1].append(roc_curve(data.y_test, y_score[:,1], pos_label=1))

        reports[estimator_name][data.name] = {
            DataSplit.TRAIN: classification_report(data.y_train, predicted_y_train, zero_division=0, output_dict=False),
            DataSplit.TEST: classification_report(data.y_test, predicted_y_test, zero_division=0, output_dict=False)
        }

        #print(classification_report(data.y_test, predicted_y_test, zero_division=0))
data_names = list(map(lambda dataset: dataset.name, datasets))  # dataset names
model_names = list(map(lambda model: model['estimator'].__class__.__name__, models))  # model names

roc_data = np.array(roc_data)
for m in range(len(models)):
    graph_classification_reports(models[m]['estimator'].__class__.__name__, roc_data[m], data_names)

for d in range(len(datasets)):
    graph_classification_reports(datasets[d].name, roc_data[:, d], model_names)



print('Done!')

for model in reports.keys():
    for data in reports[model].keys():
        print('\n')
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            print(f'{model}: {data}, {split}')
            print(reports[model][data][split])


