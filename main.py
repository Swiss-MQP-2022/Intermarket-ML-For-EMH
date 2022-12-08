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
from constants import ConsensusBaseline, CONSENSUS_BASELINES, DataSplit, Model, POLLING_RATE


def fit_single_model(model_trainer: ScikitModelTrainer, dataset: TimeSeriesDataset, report_dict: dict[str, dict], replication: int = None):
    """
    Fit a single model on the provided dataset and report results
    :param model_trainer: model trainer to use
    :param dataset: dataset to fit to
    :param report_dict: dictionary to save results in
    :param replication: replication identifier for this experiment (None indicates unreplicated)
    """
    print(f'Fitting {model_trainer.name} on {dataset.name}'
          f'{" using GridSearchCV" if model_trainer.use_grid_search else ""}'
          f' ({f"Replication {replication}, " if replication is not None else ""}PID {os.getpid()})...')

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

    print(f'Done fitting {model_trainer.name} on {dataset.name} '
          f'({f"Replication {replication}, " if replication is not None else ""}PID {os.getpid()})')


def fit_consensus_baseline(dataset_list: list[TimeSeriesDataset],
                           report_dict: dict[str, dict],
                           baseline: ConsensusBaseline,
                           replication: int = None):
    """
    Fit and record results for the Consensus Baseline (Previous Baseline is period=1)
    :param dataset_list: list of datasets to fit on
    :param report_dict: dictionary to save reports to
    :param baseline: desired baseline
    :param replication: replication identifier for this experiment (None indicates unreplicated)
    """
    print(f'Generating {baseline}{f" (Replication {replication}, PID {os.getpid()})" if replication is not None else ""}...')

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

    print(f'Done generating {baseline}{f" (Replication {replication}, PID {os.getpid()})" if replication is not None else ""}')


def start_new_model_process(model_trainer: ScikitModelTrainer,
                            dataset: TimeSeriesDataset,
                            process_list: list[mp.Process],
                            reports: dict[Model, dict[str, dict[DataSplit]]],
                            replication: int = None):
    """
    Starts a new model-fitting process
    NOTE: adds the newly created process to process_list
    :param model_trainer: ScikitModelTrainer to use
    :param dataset: dataset to fit model on
    :param process_list: list of processes to add to
    :param reports: dictionary of reports to save results in
    :param replication: replication identifier for this experiment (None indicates unreplicated)
    """
    # Create process to fit a single model
    new_process = mp.Process(target=fit_single_model,
                             args=(model_trainer, dataset, reports, replication),
                             daemon=True)
    process_list.append(new_process)  # add process to process list
    new_process.start()  # start process


def wait_for_processes(process_list: list[mp.Process], wait_threshold: int, polling_rate: int):
    """
    Wait until the number of outstanding processes is below a specified amount
    NOTE: closes any processes that terminate and removes them from process_list
    NOTE: WILL ALWAYS CHECK FOR AND CLOSE ANY DEAD PROCESSES AT LEAST ONCE
    :param process_list: list of processes to wait on
    :param wait_threshold: minimum number of processes require waiting (never waits if wait_threshold is -1)
    :param polling_rate: rate (in seconds) to poll if any processes terminated
    """
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


def run_experiment(model_params: dict[Model, dict],
                   datasets: list[TimeSeriesDataset],
                   selected_model: Model,
                   processes: int, n_jobs: int,
                   out_dir: str,
                   replication: int = None):
    """
    Run a single experiment replication
    :param model_params: model parameters to use
    :param datasets: datasets to run experiment on
    :param selected_model: specific model to test (test all if None)
    :param processes: maximum number of concurrent processes to use
    :param n_jobs: n_jobs parameter to pass to scikit-learn models
    :param out_dir: directory to save results to
    :param replication: replication identifier for this experiment (None indicates unreplicated)
    """

    print(f'Beginning experiment{f" (Replication {replication}, PID {os.getpid()})" if replication is not None else ""}...')

    # Specific model selected
    if selected_model is not None:
        if selected_model in CONSENSUS_BASELINES:  # consensus model requires manual calculation, no parameters needed
            model_params = {}
        else:  # only provide parameters for desired model
            model_params = {selected_model: model_params[selected_model]}

    # Dictionary which stores result data from experiments
    reports = {}  # Organized as reports[model name][dataset name][split]
    process_list = []  # List of processes (used for multiprocessing)

    # Model experimentation
    for model_name, model in model_params.items():  # For each model
        trainer = ScikitModelTrainer(**model, n_jobs=n_jobs, name=model_name)  # Initialize a trainer for the model
        reports[trainer.name] = mp.Manager().dict()  # Initialize dictionary for reports associated with model

        for data in datasets:  # For each dataset
            if processes is None:  # multiprocessing not requested
                fit_single_model(trainer, data, reports, replication)  # Fit a single model in the current process
            else:  # multiprocessing enabled
                wait_for_processes(process_list, processes,
                                   POLLING_RATE)  # Wait for acceptable number of running processes
                start_new_model_process(trainer, data, process_list, reports, replication)  # Start a new fitting process

    wait_for_processes(process_list, 1, POLLING_RATE)  # Wait for any still-running processes to terminate

    # Consensus baselines
    if selected_model is None:  # no model selected
        for baseline in CONSENSUS_BASELINES:  # fit all consensus baselines
            fit_consensus_baseline(datasets, reports, baseline, replication)
    elif selected_model in CONSENSUS_BASELINES:  # consensus baseline selected as model
        fit_consensus_baseline(datasets, reports, selected_model, replication)  # fit desired baseline

    # RESULT REPORTING
    results = encode_results(reports)

    # print(results)

    # Save metrics
    save_results(results, selected_model, out_dir=out_dir, prefix=replication)

    print(f'Experiment completed!{f" (Replication {replication}, PID {os.getpid()})" if replication is not None else ""}')


def get_model_trainer_params(n_jobs: int = -1):
    """
    Initialize model trainer parameters
    :param n_jobs: n_jobs parameter to pass to models which support parallel processing
    :return: dictionary of model trainer parameters
    """

    return {
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


def initialize_option_parser():
    """
    Initializes the option parser for the main script
    :return: the option parser
    """
    parser = OptionParser(option_class=OptionWithModel)
    parser.add_option('-p', '--processes',
                      action='store',
                      type='int',
                      dest='processes',
                      help='Number of processes to use PER REPLICATION. '
                           'Unlimited processes if no number is provided. '
                           '1 process per replication if option is not present. ')
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
    parser.add_option('-r', '--replications',
                      action='store',
                      type='int',
                      dest='replications',
                      help='Number of replications to perform. '
                           'Note: Each replication is given a dedicated process. '
                           'As a result, the n_jobs parameter for supported scikit-learn models will be set to 1. '
                           'Therefore combining this option with -p/--processes is recommended.')

    return parser


if __name__ == '__main__':
    # Initialize option parser for optional multiprocessing parameter
    parser = initialize_option_parser()
    options, _ = parser.parse_args()

    if options.use_uuid:
        options.out_dir += rf'_{uuid.uuid4()}'
        print(f'Unique output directory requested. Using {options.out_dir}')

    Path(options.out_dir).mkdir(parents=True, exist_ok=True)  # create output directory if it doesn't exist

    # n_jobs parameter for scikit-learn models (must be 1 when using multiprocessing)
    n_jobs = 1 if options.processes is not None or options.replications is not None else -1

    # Initialize estimators and parameters to use for experiments
    model_params = get_model_trainer_params(n_jobs)

    # Construct datasets to experiment on
    datasets = build_datasets(period=5,
                              rand_features=5,
                              test_size=0.2,
                              zero_col_thresh=0.25,
                              replace_zero=-1)

    if options.replications is None:  # no replications requested
        run_experiment(model_params, datasets, options.model, options.processes, n_jobs, options.out_dir)
    else:  # replication requested
        replication_processes = []  # list of processes for each replication
        for replication in range(options.replications):  # for each replication
            new_experiment = mp.Process(target=run_experiment,  # create a process for the replication
                                        args=(model_params, datasets, options.model, options.processes, n_jobs,
                                              options.out_dir, replication),
                                        daemon=False)
            replication_processes.append(new_experiment)  # add replication process to list
            new_experiment.start()  # run replication

        # wait for ALL replications to complete
        wait_for_processes(replication_processes, 1, POLLING_RATE)

    print('All experiments completed!\nExiting...')
