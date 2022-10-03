from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.dummy import DummyClassifier

from dataset import build_datasets
from trainer import ScikitModelTrainer, DataSplit

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

    # dict(estimator=SVC()),
    # dict(estimator=KNN()),
    # dict(estimator=LogisticRegression())
    dict(estimator=DummyClassifier(strategy='most_frequent')),
    dict(estimator=DummyClassifier(strategy='prior')),
    dict(estimator=DummyClassifier(strategy='uniform', random_state=0))
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
