# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

# def standardize(x):
#     """Standardize the original data set.
#     Using feature scaling:
#     X = (X - Xmin) / (Xmax - Xmin)
#     """    
#     print("x.min(0).shape", x.min(0).shape)
#     print("x.min(1).shape", x.min(1).shape)
#     x  = ((x.T - x.min(1)) / (x.max(1) - x.min(1))).T 
#     tx = np.hstack((np.ones((x.shape[0],1)), x))
#     return tx, x.min(1), x.max(1)


def standardize(x, minX = None, rangeX = None):
    """Standardize the original data set.
    Using feature scaling:
    X = (X - Xmin) / (Xmax - Xmin)
    """
    if minX is None:
        minX = x.min(0)

    if rangeX is None:
        rangeX = x.max(0) - x.min(0)
        rangeX[rangeX==0] = 1

    x  = ((x - minX) / rangeX) 
    # tx = np.hstack((np.ones((x.shape[0],1)), x))
    return x, minX, rangeX


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
    #create the matrix tx
    tx = np.ones((x.shape[0], degree+1))
    for i in range(x.shape[0]):
        for j in range(degree+1):
            tx[i, j] = np.power(x[i],j)
    return tx
