from optparse import OptionParser
import multiprocessing as mp

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from dataset import build_datasets
from trainer import ScikitModelTrainer, DataSplit


def fit_single_model(model_trainer, dataset):
    model_name = model_trainer.estimator.__class__.__name__
    print(f'Fitting {model_name} on {dataset.name}{" using GridSearchCV" if model_trainer.use_grid_search else ""}...')

    clf = model_trainer.train(dataset.X_train, dataset.y_train)
    predicted_y_train = clf.predict(dataset.X_train)
    predicted_y_test = clf.predict(dataset.X_test)

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

for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}

    for data in datasets:
        if options.multiprocess:
            pr.append(mp.Process(target=fit_single_model, args=(trainer, data)))
        else:
            fit_single_model(trainer, data)

[p.start() for p in pr]
[p.join() for p in pr]

print('Done!')

for model in reports.keys():
    for data in reports[model].keys():
        print('\n')
        for split in [DataSplit.TRAIN, DataSplit.TEST]:
            print(f'{model}: {data}, {split}')
            print(reports[model][data][split])
