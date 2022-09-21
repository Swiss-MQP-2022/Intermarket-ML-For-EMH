import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import utils
from timeseries_dataset import TimeSeriesDataset
from trainer import ScikitModelTrainer, DataSplit

all_data = utils.load_data()

pct_df = utils.make_pct_data(all_data['stock']['SPY.US'])[1:]

X = pct_df[:-1]
y = np.sign(pct_df['close'].to_numpy())[1:]

period = 5
features = X.shape[1]

datasets = [
    TimeSeriesDataset(X, y, period=period, name='SPY')
]

models = [
    dict(estimator=DecisionTreeClassifier(),
         param_grid=dict(splitter=['best', 'random'],
                         max_depth=[5, 10, 25, None],
                         min_samples_split=[2, 5, 10, 50],
                         min_samples_leaf=[1, 5, 10])),

    dict(estimator=SVC()),
    dict(estimator=KNN()),
    dict(estimator=LogisticRegression())
]

reports = {}

for model in models:
    trainer = ScikitModelTrainer(**model)
    estimator_name = model['estimator'].__class__.__name__
    reports[estimator_name] = {}

    for data in datasets:
        print(
            f'Fitting {estimator_name} on {data.name}{" using GridSearchCV" if "param_grid" in model.keys() else ""}...')

        clf = trainer.train(data.X_train, data.y_train)
        predicted_y_train = clf.predict(data.X_train)
        predicted_y_test = clf.predict(data.X_test)
        reports[estimator_name][data.name] = {
            DataSplit.TRAIN: classification_report(data.y_train, predicted_y_train, zero_division=0, output_dict=True),
            DataSplit.TEST: classification_report(data.y_test, predicted_y_test, zero_division=0, output_dict=True)
        }

print('Done!')
