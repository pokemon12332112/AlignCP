import numpy as np
import random


def standard_split(x, y, p=None, seed=None):
    calib_idx, val_idx = [], []
    for val in np.unique(y):
        idx = np.where(y == val)[0]
        split_value = np.max([1, int(len(idx)*p)])
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_value])
        val_idx.extend(idx[split_value:])
    assert set(calib_idx).intersection(set(val_idx)) == set(), 'Overlapping indices.'

    x_calib, y_calib = x[calib_idx], y[calib_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    return x_calib, y_calib, x_val, y_val


def balance_split(x, y, k=16, p=None, seed=None):
    y = np.int8(y)
    N = len(np.unique(y)) * k

    calib_idx, val_idx = [], []
    for val in list(np.unique(y)):
        idx = np.where(y == val)[0]
        split_value = np.max([1, round(N*p[val])]) 
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(idx, )
        calib_idx.extend(idx[:split_value])

    x_calib, y_calib = x[calib_idx], y[calib_idx]

    return x_calib, y_calib
