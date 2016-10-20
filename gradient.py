# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np
from helpers import *


def compute_gradient(y, tx, w):
    """Compute the gradient
    """
    e = y - tx.dot(w)
    n = len(e)
    grad = -(tx.T @ e) / n
    return grad


def compute_stoch_gradient(y, tx, w, batch_size):
    """Compute a stochastic gradient for batch data.
    """
    stoch_grad = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad = stoch_grad + compute_gradient(minibatch_y, minibatch_tx, w)
        
    stoch_grad = stoch_grad / batch_size
    return stoch_grad

def calculate_gradient_sigmoid(y, tx, w):
    """compute the gradient of loss with sigmoid function.
    """
    grad = tx.T @ (np.subtract(sigmoid(tx @ w),y))
    return grad

def sigmoid(t):
    """apply sigmoid function on t.
    Apply it to each row of the vecotr given in input
    """
    y = np.zeros((t.shape[0],1))
    for i in range(t.shape[0]):
        y[i] = math.exp(t[i]) / (1 + math.exp(t[i]))

    return y

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function.
    """
    sigmo = sigmoid(tx @ w)
    one_vec = np.ones((len(sigmo),1))
    minus_sigmo = (np.subtract(one_vec,sigmo))
    S_array = np.multiply(sigmo, minus_sigmo)
    S = np.diag(S_array[:,0])

    H = tx.T @ S @ tx
    return H


