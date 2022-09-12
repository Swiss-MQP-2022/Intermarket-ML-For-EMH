import numpy as np


def percent_diff(array):
    rolled = np.roll(array, -1, axis=0)
    return (rolled - array) / array
