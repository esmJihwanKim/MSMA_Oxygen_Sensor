'''
Savitsky-Golay filter
'''

from scipy import signal
from statistics import mean
from sklearn.metrics import mean_squared_error
import numpy as np
from numpy import diff


def mean_absolute_error(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    mean_diff = abs_diff.mean()
    return mean_diff


def optimize_savgol(dev):
    error = 100
    opt_window_size = 10
    opt_order = 1
    # i is the window size used for filtering
    # j is the order of fitted polynomial
    for i in range(11,100):
        for j in range(1,10):
            devs = signal.savgol_filter(dev, i, j)
            devs = np.squeeze(np.asarray(devs))
            error_try = mean_squared_error(dev, devs)
            if error > error_try:
                if np.mean(np.absolute(diff(devs))) < 0.0001:
                    error = error_try
                    opt_window_size = i
                    opt_order = j
    return [error, opt_window_size, opt_order]


def optimize_original_savgol(dev):
    error = 100
    opt_window_size = 10
    opt_order = 1
    for i in range (11,100):
        for j in range(1,10):
            devs = signal.savgol_filter(dev, i, j)
            devs = np.squeeze(np.asarray(devs))
            error_try = mean_squared_error(dev, devs)
            if error > error_try:
                if np.mean(np.absolute(diff(devs))) < 0.0001:
                    error = error_try
                    opt_window_size = i
                    opt_order = j
    return [error, opt_window_size, opt_order]


def check_for_less(data, val):
    for x in data:
        if val < x:
            return False
    return True

