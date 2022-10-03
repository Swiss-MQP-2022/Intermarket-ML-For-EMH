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
from trainer import ScikitModelTrainer
from utils import DataSplit, print_classification_reports
from classification_graphs import graph_roc


def fit_single_model(model_trainer, dataset, report_dict):
    model_name = model_trainer.estimator.__class__.__name__  # Save estimator name
    print(f'Fitting {model_name} on {dataset.name}{" using GridSearchCV" if model_trainer.use_grid_search else ""}...')

    # Train/fit provided model trainer on the provided dataset
    clf = model_trainer.train(dataset.X_train, dataset.y_train)

    predicted_y_train = clf.predict(dataset.X_train)  # Get prediction of fitted model on training set (in-sample)
    predicted_y_test = clf.predict(dataset.X_test)  # Get prediction of fitted model on test set (out-sample)
    y_score = clf.predict_proba(dataset.X_test)  # Get probabilities for predictions on test set (used for ROC)

    # Update report dictionary with results
    report_dict[model_name][dataset.name] = {
        'classification report': {
            DataSplit.TRAIN: classification_report(dataset.y_train, predicted_y_train, zero_division=0, output_dict=False),
            DataSplit.TEST: classification_report(dataset.y_test, predicted_y_test, zero_division=0, output_dict=False)
        },
        'roc': roc_curve(dataset.y_test, y_score[:, -1])
    }


if __name__ == '__main__':
    # Initialize option parser for optional multiprocessing parameter
    parser = OptionParser()
    parser.add_option('-m', '--multiprocess',
                      action='store_true',
                      default=False,
                      dest='multiprocess',
                      help='Use multiprocessing when fitting models')
    options, _ = parser.parse_args()

    # Initialize estimators and parameters to use for experiments
    models = [
        dict(estimator=DecisionTreeClassifier(),
             param_grid=dict(splitter=['best', 'random'],
                             max_depth=[5, 10, 25, None],
                             min_samples_split=[2, 5, 10, 50],
                             min_samples_leaf=[1, 5, 10])),
        dict(estimator=SVC(probability=True),
             param_grid=dict(kernel=["linear", "poly", "rbf", "sigmoid"],
                             shrinking=[True, False],
                             probability=[True, False],
                             C=[1, 4, 9, 16, 25])),
        dict(estimator=KNN(n_jobs=-1),
             param_grid=dict(n_neighbors=[5, 10, 15, 20],
                             weights=['uniform', 'distance'],
                             metric=['l1', 'l2', 'cosine', 'haversine'])),
        dict(estimator=LogisticRegression(max_iter=1000),
             param_grid=dict(penalty=['l1', 'l2'],
                             c=np.logspace(-3, 3, 7),
                             solver=['newton-cg', 'lbfgs', 'liblinear']),
             error_score=0),
        dict(estimator=DummyClassifier(strategy='prior')),
        dict(estimator=DummyClassifier(strategy='uniform', random_state=0))
    ]

    # Construct datasets to experiment on
    datasets = build_datasets(period=5,
                              brn_features=5,
                              zero_col_thresh=0.25,
                              replace_zero=-1,
                              svd_solver='full', n_components=0.95)

    pr = []  # List of processes (used for multiprocessing)
    reports = {}  # Dictionary which stores result data from experiments

    # Model experimentation
    for model in models:  # For each model
        trainer = ScikitModelTrainer(**model)  # Initialize a trainer for the model
        estimator_name = model['estimator'].__class__.__name__  # save model name
        reports[estimator_name] = mp.Manager().dict()  # Initialize dictionary for reports associated with model

        for data in datasets:  # For each dataset
            if options.multiprocess:  # Use multiprocessing if enabled
                # Create job (process) to fit a single model
                pr.append(mp.Process(target=fit_single_model, args=(trainer, data, reports)))
            else:  # Do not use multiprocessing
                # Fit a single model in the current process
                fit_single_model(trainer, data, reports)

    [p.start() for p in pr]  # Start all model-fitting jobs
    [p.join() for p in pr]  # Wait for all model-fitting jobs to complete

    # Convert reports into a multi-level dataframe of results
    results = pd.DataFrame.from_dict({(m, d): reports[m][d]
                                      for m in reports.keys()
                                      for d in reports[m].keys()},
                                     orient='index')

    print('Generating ROC graphs...')

    # Generate ROC w.r.t. model plots
    for model_name, model in tqdm(results['roc'].groupby(level=0)):
        graph_roc(f'model: {model_name}', model.to_numpy(), model.index.get_level_values(1).tolist())

    # Generate ROC w.r.t. dataset plots
    for data_name, dataset in tqdm(results['roc'].groupby(level=1)):
        graph_roc(f'dataset: {data_name}', dataset.to_numpy(), dataset.index.get_level_values(0).tolist())

    print_classification_reports(results)

    print('Done!')
