# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Grid Search
"""

import numpy as np
#import costs


def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]


# ***************************************************
# INSERT YOUR CODE HERE
# TODO: Paste your implementation of grid_search
#       here when it is done.
# ***************************************************

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    loss = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([w0[i], w1[j]])
            loss[i, j] = compute_cost(y, tx, w)
    
    return loss

def compute_cost(y, tx, w):
    """calculate the cost.

    you can calculate the cost by mse or mae.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE / MAE
    # ***************************************************
    
    # vector e
    e = compute_e(y, tx, w)
    N = compute_N(e)
    L_MSE = np.dot(np.matrix.transpose(e), e)
    L_MSE = L_MSE / (2 * N)
    
    return L_MSE

def compute_e(y, tx, w):
    return (y - np.dot(tx,w))

def compute_N(e):
    return e.shape[0]