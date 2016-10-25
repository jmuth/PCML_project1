# -*- coding: utf-8 -*-
"""Exercise 3.

Split the dataset based on the given ratio.
"""


import numpy as np


def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    index = np.arange(x.shape[0])
    shuffled_index = np.random.permutation(index)
    
    n_train = np.int(x.shape[0] * ratio)
    n_test = x.shape[0] - n_train
    
    shuffled_index_train = shuffled_index[: n_train]
    shuffled_index_test = shuffled_index[n_train : x.shape[0]]
    
    x_train = x[shuffled_index_train]
    y_train = y[shuffled_index_train]
    x_test = x[shuffled_index_test]
    y_test = y[shuffled_index_test]
    
    return x_train, y_train, x_test, y_test