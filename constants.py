from enum import Enum
from typing import Literal, get_args

import pandas as pd
from sklearn.preprocessing import FunctionTransformer


class DataSplit(str, Enum):
    TRAIN = 'Train'
    VALIDATE = 'Validation'  # unused
    TEST = 'Test'
    ALL = 'All'  # unused


class Model(str, Enum):
    DECISION_TREE = 'DecisionTree'
    RANDOM_FOREST = 'RandomForest'
    SUPPORT_VECTOR_MACHINE = 'SVM'
    K_NEAREST_NEIGHBORS = 'KNN'
    LOGISTIC_REGRESSION = 'LogisticRegression'
    RANDOM_BASELINE = 'RandomBaseline'
    CONSTANT_BASELINE = 'ConstantBaseline'
    PREVIOUS_BASELINE = 'PreviousBaseline'
    CONSENSUS_BASELINE = 'ConsensusBaseline'


DataDict = dict[str, dict[str, pd.DataFrame]]
AssetID = tuple[str, str]
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

DUMMY_SCALER = FunctionTransformer(lambda x: x)

METRICS = {
    'Accuracy': 'accuracy',
    'Weighted F1': ('weighted avg', 'f1-score'),
    'Macro F1': ('macro avg', 'f1-score')
}

POLLING_RATE = 30  # Rate in seconds to poll changes in process status
