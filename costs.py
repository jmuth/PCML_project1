# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np
import math


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_rmse(e):
    """Calculate the rmse for vector e"""
    return np.sqrt(2* calculate_mse(e))

def calculate_negative_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # ***************************************************
    L = 0
    # print("debug loss")
    # print("tx", tx.shape)
    # print("w", w.shape)
    for i in range(tx.shape[0]):
        # because of overflow error I'll do some trick here. Instead of computing
        # the ln(1 + exp(x)), I'll test if x > 700, and if, just discard the +1 and
        # only use x (very, very small error)
        x = tx[i] @ w
        if (x < 700):
            exp = math.exp(tx[i] @ w)
            log_n = np.log(1 + exp)
        else:
            log_n = x
        
        yxw = y[i] * tx[i].T @ w
        L += log_n - yxw
        
    return L

def calculate_loss(y, tx, w, method="log"):
    """Calculate the loss.

    Calculate loss using MSE, MAE, RMSE or Neg Log Likelihood.
    (The last choice is our for this project)
    """
    e = y - tx.dot(w)
    # return calculate_mse(e)
    # return calculate_mae(e)
    # return calculate_rmse(e)
    if(method=="log"):
        return calculate_negative_log_likelihood(y, tx, w)
    elif(method=="rmse"):
        return calculate_rmse(e)



