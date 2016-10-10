# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
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


def grid_search(y, tx, w0, w1):
    """Algorithm for grid search."""
    losses = np.zeros((len(w0), len(w1)))
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss for each combination of w0 and w1.
    # ***************************************************
    
    for i in range(len(w0)):
        for j in range(len(w1)):
            w = np.array([w0[i], w1[j]])
            losses[i, j] = compute_cost(y, tx, w)
    
    return losses

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