import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import utils
from timeseries_dataset import TimeSeriesDataLoader


# Load Data
spy = pd.read_csv(r'./data/stock/SPY.US.csv').set_index('date')  # Load data from file
spy = utils.get_nonempty_float_columns(spy).dropna()  # filter to numeric columns. Drop NaNs

X_0 = spy.iloc[0]  # record initial raw X values

# brn = utils.generate_brownian_motion(len(spy), len(spy.columns), initial=X_0.to_numpy())
# print(len(spy))

pct_df = spy.pct_change()[1:]  # Compute percent change
pct_df = utils.remove_outliers(pct_df)

X = pct_df.to_numpy()[:-1]

y = np.sign(pct_df['close'].to_numpy())[1:]

period = 5
features = X.shape[1]
dataloader = TimeSeriesDataLoader(X, y, period=period, test_size=.20)

model = DecisionTreeClassifier()
model.fit(dataloader.X_train.reshape(-1, period * features), dataloader.y_train)

plt.figure(figsize=(20, 15))
tree.plot_tree(model, ax=plt.gca(), fontsize=10)
plt.tight_layout()
plt.show()
print('Feature importance:', model.feature_importances_)

predicted_y_train = model.predict(dataloader.X_train.reshape(-1, period * features))
print('Train report:')
print(classification_report(dataloader.y_train, predicted_y_train))

predicted_y_test = model.predict(dataloader.X_test.reshape(-1, period * features))
print('Test report:')
print(classification_report(dataloader.y_test, predicted_y_test))
