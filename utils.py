import numpy as np


def prices_to_percent_change(array):
    return np.diff(array) / array
