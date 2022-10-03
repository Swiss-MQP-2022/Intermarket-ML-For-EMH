from optparse import OptionParser
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.dummy import DummyClassifier

from dataset import build_datasets
from trainer import ScikitModelTrainer, DataSplit
from classification_graphs import graph_roc


def fit_single_model(model_trainer, dataset, report_dict):
    model_name = model_trainer.estimator.__class__.__name__
    print(f'Fitting {model_name} on {dataset.name}{" using GridSearchCV" if model_trainer.use_grid_search else ""}...')

    clf = model_trainer.train(dataset.X_train, dataset.y_train)
    predicted_y_train = clf.predict(dataset.X_train)
    predicted_y_test = clf.predict(dataset.X_test)
    y_score = clf.predict_proba(dataset.X_test)

    # print(f'{estimator_name} on {dataset.name} results:')
    # print(classification_report(dataset.y_test, predicted_y_test, zero_division=0))

    report_dict[model_name][dataset.name] = {
        'classification report': {
            DataSplit.TRAIN: classification_report(dataset.y_train, predicted_y_train, zero_division=0, output_dict=False),
            DataSplit.TEST: classification_report(dataset.y_test, predicted_y_test, zero_division=0, output_dict=False)
        },
        'roc': roc_curve(dataset.y_test, y_score[:, -1])
    }


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-m', '--multiprocess',
                      action='store_true',
                      default=False,
                      dest='multiprocess',
                      help='Use multiprocessing when fitting models')
    options, _ = parser.parse_args()

    models = [
        dict(estimator=DecisionTreeClassifier(),
             param_grid=dict(splitter=['best', 'random'],
                             max_depth=[5, 10, 25, None],
                             min_samples_split=[2, 5, 10, 50],
                             min_samples_leaf=[1, 5, 10])),
        dict(estimator=SVC(probability=True)),
        dict(estimator=KNN()),
        dict(estimator=LogisticRegression(max_iter=1000)),
        dict(estimator=DummyClassifier(strategy='prior')),
        dict(estimator=DummyClassifier(strategy='uniform', random_state=0))
    ]

    datasets = build_datasets(period=5,
                              brn_features=5,
                              zero_col_thresh=0.25,
                              replace_zero=-1,
                              svd_solver='full', n_components=0.95)

    pr = []
    reports = {}

    for model in models:
        trainer = ScikitModelTrainer(**model)
        estimator_name = model['estimator'].__class__.__name__
        reports[estimator_name] = mp.Manager().dict()

        for data in datasets:
            if options.multiprocess:
                pr.append(mp.Process(target=fit_single_model, args=(trainer, data, reports)))
            else:
                fit_single_model(trainer, data, reports)

    [p.start() for p in pr]
    [p.join() for p in pr]

    reports = pd.DataFrame.from_dict({(m, d): reports[m][d]
                                      for m in reports.keys()
                                      for d in reports[m].keys()},
                                     orient='index')

    print('Generating ROC graphs...')

    for model_name, model in tqdm(reports['roc'].groupby(level=0)):
        graph_roc(f'model: {model_name}', model.to_numpy(), model.index.get_level_values(1).tolist())

    for data_name, dataset in tqdm(reports['roc'].groupby(level=1)):
        graph_roc(f'dataset: {data_name}', dataset.to_numpy(), dataset.index.get_level_values(0).tolist())

    print('Printing classification reports...')

    for model_name, model in reports['classification report'].groupby(level=0):
        for data_name, clf_report in model.droplevel(0).items():
            for split in [DataSplit.TRAIN, DataSplit.TEST]:
                print(f'{model_name}: {data_name}, {split}')
                print(clf_report[split])

    print('Done!')
