import numpy as np


def percent_diff(array):
    rolled = np.roll(array, 1, axis=0)
    array = (array - rolled) / array
    array = array[1:, ]
    return array