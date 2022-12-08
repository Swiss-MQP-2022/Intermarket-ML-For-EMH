from optparse import OptionParser
import multiprocessing as mp
from pathlib import Path
from time import sleep
import uuid
import os

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

from dataset import build_datasets, TimeSeriesDataset
from trainer import ScikitModelTrainer
from utils import OptionWithModel, align_data, compute_consensus, encode_results, save_results
from constants import ConsensusBaseline, CONSENSUS_BASELINES, DataSplit, Model

POLLING_RATE = 30  # Rate in seconds to poll changes in process status


def fit_single_model(model_trainer: ScikitModelTrainer, dataset: TimeSeriesDataset, report_dict: dict[str, dict]):
    """
    Fit a single model on the provided dataset and report results
    :param model_trainer: model trainer to use
    :param dataset: dataset to fit to
    :param report_dict: dictionary to save results to
    """
    print(f'Fitting {model_trainer.name} on {dataset.name}'
          f'{" using GridSearchCV" if model_trainer.use_grid_search else ""}'
          f' (PID {os.getpid()})...')

    # Train/fit provided model trainer on the provided dataset
    clf = model_trainer.train(dataset.X_train, dataset.y_train)

    predicted_y_train = clf.predict(dataset.X_train)  # Get prediction of fitted model on training set (in-sample)
    predicted_y_test = clf.predict(dataset.X_test)  # Get prediction of fitted model on test set (out-sample)

    # Update report dictionary with results
    report_dict[model_trainer.name][dataset.name] = {
        DataSplit.TRAIN: classification_report(dataset.y_train, predicted_y_train,
                                               zero_division=0, output_dict=True),
        DataSplit.TEST: classification_report(dataset.y_test, predicted_y_test,
                                              zero_division=0, output_dict=True)
    }

    print(f'Done fitting {model_trainer.name} on {dataset.name} (PID {os.getpid()})')


def fit_consensus_baseline(dataset_list: list[TimeSeriesDataset],
                           report_dict: dict[str, dict],
                           baseline: ConsensusBaseline):
    """
    Fit and record results for the Consensus Baseline (Previous Baseline is period=1)
    :param dataset_list: list of datasets to fit on
    :param report_dict: dictionary to save reports to
    :param baseline: desired baseline
    """
    print(f'Generating {baseline}...')

    report_dict[baseline] = {}
    for dataset in dataset_list:
        # set period based on desired baseline (Consensus uses existing dataset's period, Previous uses 1)
        period = dataset.period if baseline == 'ConsensusBaseline' else 1
        # generate consensus predictions
        train_consensus = compute_consensus(dataset.y_train.shift(1).iloc[1:], period)
        test_consensus = compute_consensus(dataset.y_test.shift(1).iloc[1:], period)

        report_dict[baseline][dataset.name] = {
            DataSplit.TRAIN: classification_report(*align_data(train_consensus, dataset.y_train),
                                                   zero_division=0, output_dict=True),
            DataSplit.TEST: classification_report(*align_data(test_consensus, dataset.y_test),
                                                  zero_division=0, output_dict=True)
        }

    print(f'Done generating {baseline}')


def wait_for_processes(wait_threshold: int, polling_rate: int):
    """
    Wait until the number of outstanding processes is below a specified amount
    NOTE: closes any processes that terminate and removes them from process_list
    NOTE: WILL ALWAYS CHECK FOR AND CLOSE ANY DEAD PROCESSES AT LEAST ONCE
    :param wait_threshold: minimum number of processes require waiting (never waits if wait_threshold is -1)
    :param polling_rate: rate (in seconds) to poll if any processes terminated
    """
    global process_list

    wait = True  # flag to continue waiting. Used to emulate a do-while loop
    while wait:  # While we need to wait for processes to finish
        for process in process_list:  # for each remaining process
            if not process.is_alive():  # if process finished
                print(f'Closing process {process.pid}')
                process.close()  # release resources
                process_list.remove(process)  # remove process from process list

        # update wait flag based on number of active processes (always False if threshold is -1)
        wait = len(process_list) >= wait_threshold != -1
        if wait:  # sleep if we still need to wait
            sleep(polling_rate)


def start_new_model_process(model_trainer: ScikitModelTrainer, dataset: TimeSeriesDataset):
    """
    Starts a new model-fitting process
    NOTE: adds the newly created process to process_list
    :param model_trainer: ScikitModelTrainer to use
    :param dataset: dataset to fit model on
    """
    global process_list, reports

    # Create process to fit a single model
    new_process = mp.Process(target=fit_single_model, args=(model_trainer, dataset, reports), daemon=True)
    process_list.append(new_process)  # add process to process list
    new_process.start()  # start process


def initialize_option_parser():
    """
    Initializes the option parser for the main script
    :return: parser
    """
    parser = OptionParser(option_class=OptionWithModel)
    parser.add_option('-p', '--processes',
                      action='store',
                      type='int',
                      dest='processes',
                      help='Use multiprocessing when fitting models')
    parser.add_option('-m', '--model',
                      action='store',
                      type='model_name',
                      dest='model',
                      help='Singular model to train')
    parser.add_option('-o', '--out-dir',
                      action='store',
                      type='str',
                      default='./out',
                      dest='out_dir',
                      help='Path to directory to save output files to')
    parser.add_option('-u', '--use-uuid',
                      action='store_true',
                      default=False,
                      dest='use_uuid',
                      help='Appends a unique identifier the output directory')

    return parser


def get_model_trainer_params(model: Model = None, n_jobs: int = -1):
    """
    Initialize model trainer parameters
    :param model: desired model to get parameters for. Provides all models of None, empty dict if in CONSENSUS_BASELINES
    :param n_jobs: n_jobs parameter to pass to models which support parallel processing
    :return: dictionary of model trainer parameters
    """

    param_dict = {
        Model.DECISION_TREE: dict(estimator=DecisionTreeClassifier(),
                                  param_grid=dict(splitter=['best', 'random'],
                                                  max_depth=[5, 10, 25, None],
                                                  min_samples_split=[2, 5, 10, 50],
                                                  min_samples_leaf=[1, 5, 10])),
        Model.RANDOM_FOREST: dict(estimator=RandomForestClassifier(n_jobs=n_jobs),
                                  param_grid=dict(n_estimators=[50, 100, 500],
                                                  criterion=['gini', 'entropy'],
                                                  max_depth=[5, 10, 25, None],
                                                  min_samples_split=[2, 5, 10, 50],
                                                  min_samples_leaf=[1, 5, 10])),
        Model.SUPPORT_VECTOR_MACHINE: dict(estimator=LinearSVC(max_iter=1e6),
                                           param_grid=dict(penalty=['l1', 'l2'],
                                                           C=[1, 4, 9, 16, 25],
                                                           loss=['hinge', 'squared_hinge']),
                                           error_score=0),
        Model.K_NEAREST_NEIGHBORS: dict(estimator=KNN(n_jobs=n_jobs),
                                        param_grid=dict(n_neighbors=[5, 10, 15, 20],
                                                        weights=['uniform', 'distance'],
                                                        metric=['l1', 'l2', 'cosine'])),
        Model.LOGISTIC_REGRESSION: dict(estimator=LogisticRegression(max_iter=1e4),
                                        param_grid=dict(penalty=['l1', 'l2'],
                                                        C=np.logspace(-3, 3, 7),
                                                        solver=['newton-cg', 'lbfgs', 'liblinear']),
                                        error_score=0),
        Model.CONSTANT_BASELINE: dict(estimator=DummyClassifier(strategy='prior')),
        Model.RANDOM_BASELINE: dict(estimator=DummyClassifier(strategy='uniform', random_state=0)),
    }

    # Specific model selected
    if model is not None:
        if model in CONSENSUS_BASELINES:  # consensus model requires manual calculation, no parameters needed
            return {}
        else:  # only provide parameters for desired model
            return {model: param_dict[model]}
    else:  # provide parameters for all models
        return param_dict


if __name__ == '__main__':
    # Initialize option parser for optional multiprocessing parameter
    parser = initialize_option_parser()
    options, _ = parser.parse_args()

    if options.use_uuid:
        options.out_dir += rf'_{uuid.uuid4()}'

    Path(options.out_dir).mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

    # n_jobs parameter sklearn (must be 1 when using multiprocessing)
    n_jobs = 1 if options.processes is not None else -1

    # Initialize estimators and parameters to use for experiments
    models = get_model_trainer_params(options.model, n_jobs)

    # Construct datasets to experiment on
    datasets = build_datasets(period=5,
                              rand_features=5,
                              test_size=0.2,
                              zero_col_thresh=0.25,
                              replace_zero=-1)

    # Dictionary which stores result data from experiments
    reports = {}  # Organized as reports[model name][dataset name][split][report type]
    process_list = []  # List of processes (used for multiprocessing)

    # Model experimentation
    for model_name, model in models.items():  # For each model
        trainer = ScikitModelTrainer(**model, n_jobs=n_jobs, name=model_name)  # Initialize a trainer for the model
        reports[trainer.name] = mp.Manager().dict()  # Initialize dictionary for reports associated with model

        for data in datasets:  # For each dataset
            if options.processes is not None:  # Use multiprocessing if enabled
                wait_for_processes(options.processes, POLLING_RATE)  # Wait for acceptable number of running processes
                start_new_model_process(trainer, data)  # Start a new fitting process
            else:  # Do not use multiprocessing
                fit_single_model(trainer, data, reports)  # Fit a single model in the current process

    wait_for_processes(1, POLLING_RATE)  # Wait for any still-running processes to terminate

    # Consensus baselines
    if options.model is None:  # no model selected
        for baseline in CONSENSUS_BASELINES:  # fit all consensus baselines
            fit_consensus_baseline(datasets, reports, baseline)
    elif options.model in CONSENSUS_BASELINES:  # consensus baseline selected as model
        fit_consensus_baseline(datasets, reports, options.model)  # fit desired baseline

    # RESULT REPORTING
    results = encode_results(reports)

    print(results)

    # Save metrics
    save_results(results, options.model, out_dir=options.out_dir)

    print('Done!')
