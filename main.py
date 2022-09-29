import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve

from dataset import build_datasets
from trainer import ScikitModelTrainer, DataSplit
from classification_graphs import graph_classification_reports

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

        roc_data[-1].append(roc_curve(data.y_test, y_score[:, -1]))

        reports[estimator_name][data.name] = {
            DataSplit.TRAIN: classification_report(data.y_train, predicted_y_train, zero_division=0, output_dict=False),
            DataSplit.TEST: classification_report(data.y_test, predicted_y_test, zero_division=0, output_dict=False)
        }

roc_data = np.array(roc_data, dtype='object')

dataset_names = list(map(lambda dataset: dataset.name, datasets))  # dataset names
model_names = list(map(lambda model: model['estimator'].__class__.__name__, models))  # model names

for m in range(len(models)):
    graph_classification_reports(f'model: {model_names[m]}', roc_data[m], dataset_names)

for d in range(len(datasets)):
    graph_classification_reports(f'dataset: {dataset_names[d]}', roc_data[:, d], model_names)


print('Done!')

for model in reports.keys():
    for data in reports[model].keys():
        print('\n')
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            print(f'{model}: {data}, {split}')
            print(reports[model][data][split])
