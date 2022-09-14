from sklearn.metrics import classification_report
import numpy as np

def pct_to_cumulative(data, initial):
    return (data + 1).cumprod() * initial


def forecast_classification_report(pred, truth):
    pred_direction = np.sign(pred)
    truth_direction = np.sign(truth)

    return classification_report(pred_direction, truth_direction)