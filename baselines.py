import numpy as np
import matplotlib.pyplot as plt
from dataset import build_datasets


def naive_method():
    datasets = build_datasets(period=5,
                              brn_features=5,
                              zero_col_thresh=0.25,
                              replace_zero=-1,
                              svd_solver='full', n_components=0.95)

    for data in datasets:

        dd= np.asarray(data.y)
        y_hat = data.y_test.copy()
        y_hat['naive'] = dd[-1]
        plt.figure(figsize=(12,8))
        plt.plot(data.y_train.index, data.y_train['Close'], label='Train')
        plt.plot(data.y_test.index,data.y_test['Close'], label='Test')
        plt.plot(y_hat.index,y_hat['naive'], label='Naive Forecast')
        plt.legend(loc='best')
        plt.title("Naive Forecast")
        plt.show()

if __name__ == "__main__":
    naive_method()