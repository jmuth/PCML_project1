# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np


def compute_gradient(y, tx, w):
    """Compute the gradient
    """
    e = np.array(compute_e(y, tx, w))
    n = len(e)
    tx_transposed = np.transpose(tx)
    grad = np.dot(tx_transposed, e)
    grad = - grad / n
    
    return grad


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient for batch data."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation.It's same as the gradient descent.
    # ***************************************************
    stoch_grad = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w)
        
    stoch_grad = stoch_grad / batch_size
    return stoch_grad
