from enum import Enum
from typing import Literal, get_args

import numpy as np
import pandas as pd


class DataSplit(str, Enum):
    TRAIN = 'Train'
    VALIDATE = 'Validation'  # unused
    TEST = 'Test'
    ALL = 'All'  # unused


class Model(str, Enum):
    DECISION_TREE = 'DecisionTree'
    RANDOM_FOREST = 'RandomForest'
    LOGISTIC_REGRESSION = 'LogisticRegression'
    SUPPORT_VECTOR_MACHINE = 'Linear SVM'
    K_NEAREST_NEIGHBORS = 'KNN'
    RANDOM_BASELINE = 'RandomBaseline'
    CONSTANT_BASELINE = 'ConstantBaseline'
    PREVIOUS_BASELINE = 'PreviousBaseline'
    CONSENSUS_BASELINE = 'ConsensusBaseline'


class Metric(str, Enum):
    ACCURACY = 'Accuracy'
    MACRO_F1 = 'Macro F1'
    WEIGHTED_F1 = 'Weighted F1'
    ROC_AUC = 'ROC AUC'


DataDict = dict[str, dict[str, pd.DataFrame]]
AssetID = tuple[str, str]
ReportDict = dict[Model, dict[str, dict[DataSplit, dict[Metric, np.float]]]]
ConsensusBaseline = Literal[Model.PREVIOUS_BASELINE, Model.CONSENSUS_BASELINE]

CONSENSUS_BASELINES = get_args(ConsensusBaseline)

DATASET_SYMBOLS = {
    'Forex': [('forex', 'USDGBP.FOREX'),
              ('forex', 'USDEUR.FOREX'),
              ('forex', 'USDCAD.FOREX'),
              ('forex', 'USDJPY.FOREX'),
              ('forex', 'EURGBP.FOREX')],

    'Bond': [('bond', 'US10Y.GBOND'),
             ('bond', 'US5Y.GBOND'),
             ('bond', 'UK5Y.GBOND'),
             ('bond', 'JP5Y.GBOND'),
             ('future', 'US.COMM')],

    'Index Futures': [('future', 'ES.COMM'),
                      ('future', 'NK.COMM'),
                      ('future', 'HSI.COMM'),
                      ('future', 'FESX.COMM'),
                      ('future', 'VIX.COMM')],

    'Commodities Futures': [('future', 'GC.COMM'),
                            ('future', 'NG.COMM'),
                            ('future', 'ZC.COMM'),
                            ('future', 'ZS.COMM')]
}

CLR_KEYS = {
    Metric.ACCURACY: 'accuracy',
    Metric.MACRO_F1: ('macro avg', 'f1-score'),
    Metric.WEIGHTED_F1: ('weighted avg', 'f1-score'),
    Metric.ROC_AUC: None
}

POLLING_RATE = 30  # Rate in seconds to poll changes in process status
