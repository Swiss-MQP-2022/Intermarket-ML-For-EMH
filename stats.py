import pandas as pd

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot

import matplotlib.pyplot as plt

from constants import DataSplit, Model


def run_test(data: pd.DataFrame, model: Model):
    df = data.loc[data[DataSplit.TEST] & ~data['Random'] & data[model]]
    df = df.drop(columns=[m.value for m in Model] + [DataSplit.TEST, 'SPY', 'Random'])
    df = df.replace({True: 1, False: -1})

    print(df)

    model = ols('accuracy ~ forex * bond * index_futures * commodities_futures', data=df).fit()
    print(model.summary())

    effects = 2*model.params.drop(labels='Intercept')
    print(effects)

    sm.qqplot(effects, line='r')
    plt.show()

    # interaction_plot(effects)
    # plt.show()

    # aov = sm.stats.anova_lm(model, typ=2)
    # print(aov)
