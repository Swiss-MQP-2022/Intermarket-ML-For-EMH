from enum import Enum
from typing import Literal, get_args

import pandas as pd
from sklearn.preprocessing import FunctionTransformer


class DataSplit(Enum):
    TRAIN = 'train'
    VALIDATE = 'validation'
    TEST = 'test'
    ALL = 'ALL'


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
    'forex': [('forex', 'USDGBP.FOREX'),
              ('forex', 'USDEUR.FOREX'),
              ('forex', 'USDCAD.FOREX'),
              ('forex', 'USDJPY.FOREX'),
              ('forex', 'EURGBP.FOREX')],

    'bond': [('bond', 'US10Y.GBOND'),
             ('bond', 'US5Y.GBOND'),
             ('bond', 'UK5Y.GBOND'),
             ('bond', 'JP5Y.GBOND'),
             ('future', 'US.COMM')],

    'index-futures': [('future', 'ES.COMM'),
                      ('future', 'NK.COMM'),
                      ('future', 'HSI.COMM'),
                      ('future', 'FESX.COMM'),
                      ('future', 'VIX.COMM')],

    'commodities-futures': [('future', 'GC.COMM'),
                            ('future', 'NG.COMM'),
                            ('future', 'ZC.COMM'),
                            ('future', 'ZS.COMM')]
}

DUMMY_SCALER = FunctionTransformer(lambda x: x)

METRICS = {
    'accuracy': 'accuracy',
    'weighted f1': ('weighted avg', 'f1-score'),
    'macro f1': ('macro avg', 'f1-score')
}
