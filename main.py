from optparse import OptionParser
import multiprocessing as mp
from time import sleep

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.calibration import CalibratedClassifierCV

from dataset import build_datasets
from trainer import ScikitModelTrainer
from utils import DataSplit, print_classification_reports
from out_functions import graph_all_roc, save_metrics


def fit_single_model(model_trainer, dataset, report_dict):
    print(f'Fitting {model_trainer.name} on {dataset.name}{" using GridSearchCV" if model_trainer.use_grid_search else ""}...')

    # Train/fit provided model trainer on the provided dataset
    clf = model_trainer.train(dataset.X_train, dataset.y_train)
    if model_trainer.name == 'SVC':
        clf = CalibratedClassifierCV(clf, cv='prefit')
        clf.fit(dataset.X_train, dataset.y_train)

    predicted_y_train = clf.predict(dataset.X_train)  # Get prediction of fitted model on training set (in-sample)
    predicted_y_test = clf.predict(dataset.X_test)  # Get prediction of fitted model on test set (out-sample)
    y_train_score = clf.predict_proba(dataset.X_train)  # Get probabilities for predictions on training set (for ROC)
    y_test_score = clf.predict_proba(dataset.X_test)  # Get probabilities for predictions on test set (for ROC)

    # Update report dictionary with results
    report_dict[model_trainer.name][dataset.name] = {
        'classification report': {
            DataSplit.TRAIN: classification_report(dataset.y_train, predicted_y_train, zero_division=0,
                                                   output_dict=True),
            DataSplit.TEST: classification_report(dataset.y_test, predicted_y_test, zero_division=0, output_dict=True)
        },
        'roc': {
            DataSplit.TRAIN: roc_curve(dataset.y_train, y_train_score[:, -1]),
            DataSplit.TEST: roc_curve(dataset.y_test, y_test_score[:, -1])
        }
    }

    print(f'Done fitting {model_trainer.name} on {dataset.name}')


if __name__ == '__main__':
    # Initialize option parser for optional multiprocessing parameter
    parser = OptionParser()
    parser.add_option('-p', '--processes',
                      action='store',
                      type='int',
                      dest='multiprocess',
                      help='Use multiprocessing when fitting models')
    parser.add_option('-m', '--model',
                      action='store',
                      type='str',
                      dest='model',
                      help='Singular model to train')
    options, _ = parser.parse_args()

    # n_jobs parameter for GridSearch (must be 1 with multiprocessing)
    n_jobs = 1 if options.multiprocess is not None else -1

    # Initialize estimators and parameters to use for experiments
    models = {
        'DecisionTree': dict(estimator=DecisionTreeClassifier(),
                             param_grid=dict(splitter=['best', 'random'],
                                             max_depth=[5, 10, 25, None],
                                             min_samples_split=[2, 5, 10, 50],
                                             min_samples_leaf=[1, 5, 10])),
        'RandomForest': dict(estimator=RandomForestClassifier(n_jobs=n_jobs),
                             param_grid=dict(n_estimators=[50, 100, 500],
                                             criterion=['gini', 'entropy'],
                                             max_depth=[5, 10, 25, None],
                                             min_samples_split=[2, 5, 10, 50],
                                             min_samples_leaf=[1, 5, 10])),
        'SVC': dict(estimator=LinearSVC(max_iter=1e6),
                    param_grid=dict(penalty=['l1', 'l2'],
                                    C=[1, 4, 9, 16, 25],
                                    loss=['hinge', 'squared_hinge']),
                    error_score=0),
        'KNN': dict(estimator=KNN(n_jobs=-1),
                    param_grid=dict(n_neighbors=[5, 10, 15, 20],
                                    weights=['uniform', 'distance'],
                                    metric=['l1', 'l2', 'cosine'])),
        'LogisticRegression': dict(estimator=LogisticRegression(max_iter=1e4),
                                   param_grid=dict(penalty=['l1', 'l2'],
                                                   C=np.logspace(-3, 3, 7),
                                                   solver=['newton-cg', 'lbfgs', 'liblinear']),
                                   error_score=0),
        'PriorBaseline': dict(estimator=DummyClassifier(strategy='prior')),
        'RandomBaseline': dict(estimator=DummyClassifier(strategy='uniform', random_state=0))
    }

    if options.model is not None:
        models = {model_name: model for model_name, model in models.items() if model_name == options.model}

    # Construct datasets to experiment on
    datasets = build_datasets(period=5,
                              brn_features=5,
                              zero_col_thresh=0.25,
                              replace_zero=-1,
                              svd_solver='full', n_components=0.95)

    pr = []  # List of processes (used for multiprocessing)
    reports = {}  # Dictionary which stores result data from experiments

    # Model experimentation
    for model_name, model in models.items():  # For each model
        trainer = ScikitModelTrainer(**model, n_jobs=n_jobs, name=model_name)  # Initialize a trainer for the model
        reports[trainer.name] = mp.Manager().dict()  # Initialize dictionary for reports associated with model

        for data in datasets:  # For each dataset
            if options.multiprocess is not None:  # Use multiprocessing if enabled
                while len(mp.active_children()) > options.multiprocess:  # Active processes is above process limit
                    sleep(5)  # Sleep before checking again if a job has finished
                # Create job (process) to fit a single model
                new_process = mp.Process(target=fit_single_model, args=(trainer, data, reports), daemon=True)
                pr.append(new_process)
                new_process.start()  # Start job
            else:  # Do not use multiprocessing
                fit_single_model(trainer, data, reports)  # Fit a single model in the current process

    [p.join() for p in pr]  # Wait for all model-fitting jobs to complete

    # Convert reports into a multi-level dataframe of results
    results = pd.DataFrame.from_dict({(m, d, r): reports[m][d][r]
                                      for m in reports.keys()
                                      for d in reports[m].keys()
                                      for r in reports[m][d].keys()},
                                     orient='index')
    results = results.unstack().swaplevel(0, 1, axis=1)  # Reorganize MultiIndexes
    # Results is a DataFrame with two index levels (model, dataset) and two column levels (report type, data split)

    # Save metrics to CSVs
    save_metrics(results, model_name=options.model if options.model is not None else '')

    # Generate ROC graphs
    graph_all_roc(results)

    # Print classification reports for all model-dataset pairs
    print_classification_reports(results)

    print('Done!')
