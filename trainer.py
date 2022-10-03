from typing import Union

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, BaseCrossValidator

from utils import Estimator


class ScikitModelTrainer:
    def __init__(self,
                 estimator: Estimator,
                 param_grid: dict[str, any] = None,
                 scoring: str = 'f1_weighted',  # Currently forcing string-specified scorers only
                 n_jobs: int = -1,
                 cv: Union[int, BaseCrossValidator] = 5,
                 name = None,
                 **gs_kws: dict[str, any]):
        """
        :param estimator: Scikit-Learn estimator to fit
        :param param_grid: parameter grid to search using GridSearchCV. Fit estimator directly if None (default)
        :param scoring: scoring technique to use in GridSearchCV
        :param n_jobs: jobs to use in GridSearchCV
        :param cv: cross-validator to use in GridSearchCV or number of folds to use with TimeSeriesSplit if an integer
        :param gs_kws: additional keyword arguments to pass to GridSearchCV
        """
        self.estimator = estimator
        self.use_grid_search = param_grid is not None
        self.name = name if name is not None else estimator.__class__.__name__

        if isinstance(cv, int):
            cv = TimeSeriesSplit(n_splits=cv)

        if self.use_grid_search:
            self.gscv = GridSearchCV(estimator=self.estimator,
                                     scoring=scoring,
                                     param_grid=param_grid,
                                     n_jobs=n_jobs,
                                     cv=cv,
                                     refit=True,
                                     **gs_kws)

    def train(self, X, y):
        """
        Fits the provided data to the trainer's estimator. Uses GridSearchCV if available
        :param X: input data
        :param y: target data
        :return: fitted estimator
        """
        if self.use_grid_search:
            self.gscv.fit(X, y)
            self.estimator = self.gscv.best_estimator_
        else:
            self.estimator.fit(X, y)

        return self.estimator
