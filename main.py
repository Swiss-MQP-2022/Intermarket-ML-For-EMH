from optparse import OptionParser
import multiprocessing as mp

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve

from dataset import build_datasets
from trainer import ScikitModelTrainer, DataSplit
from classification_graphs import graph_roc


def fit_single_model(model_trainer, dataset):
    model_name = model_trainer.estimator.__class__.__name__
    print(f'Fitting {model_name} on {dataset.name}{" using GridSearchCV" if model_trainer.use_grid_search else ""}...')

    clf = model_trainer.train(dataset.X_train, dataset.y_train)
    predicted_y_train = clf.predict(dataset.X_train)
    predicted_y_test = clf.predict(dataset.X_test)
    y_score = clf.predict_proba(data.X_test)

    roc_data[-1].append(roc_curve(data.y_test, y_score[:, -1]))

    print(f'{estimator_name} on {dataset.name} results:')
    print(classification_report(dataset.y_test, predicted_y_test, zero_division=0))

    reports[model_name][dataset.name] = {
        DataSplit.TRAIN: classification_report(dataset.y_train, predicted_y_train, zero_division=0, output_dict=True),
        DataSplit.TEST: classification_report(dataset.y_test, predicted_y_test, zero_division=0, output_dict=True)
    }


parser = OptionParser()
parser.add_option('-m', '--multiprocess',
                  action='store_true',
                  default=False,
                  dest='multiprocess',
                  help='Use multiprocessing when fitting models')
options, _ = parser.parse_args()

datasets = build_datasets(period=5,
                          brn_features=5,
                          zero_col_thresh=0.25,
                          replace_zero=-1,
                          svd_solver='full', n_components=0.95)

models = [
    dict(estimator=DecisionTreeClassifier(),
         param_grid=dict(splitter=['best', 'random'],
                         max_depth=[5, 10, 25, None],
                         min_samples_split=[2, 5, 10, 50],
                         min_samples_leaf=[1, 5, 10])),
    dict(estimator=SVC()),
    dict(estimator=KNN()),
    dict(estimator=LogisticRegression(max_iter=1000))
]

pr = []
reports = {}
roc_data = []

for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}
    roc_data.append([])

    for data in datasets:
        if options.multiprocess:
            pr.append(mp.Process(target=fit_single_model, args=(trainer, data)))
        else:
            fit_single_model(trainer, data)

[p.start() for p in pr]
[p.join() for p in pr]

roc_data = np.array(roc_data, dtype='object')

dataset_names = list(map(lambda dataset: dataset.name, datasets))  # dataset names
model_names = list(map(lambda model: model['estimator'].__class__.__name__, models))  # model names

for m in range(len(models)):
    graph_roc(f'model: {model_names[m]}', roc_data[m], dataset_names)

for d in range(len(datasets)):
    graph_roc(f'dataset: {dataset_names[d]}', roc_data[:, d], model_names)


print('Done!')

for model in reports.keys():
    for data in reports[model].keys():
        print('\n')
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            print(f'{model}: {data}, {split}')
            print(reports[model][data][split])
