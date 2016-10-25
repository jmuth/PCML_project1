# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_rmse(e):
    """Calculate the rmse for vector e"""
    return np.sqrt(2* calculate_mse(e))


def calculate_log(y, tx, w):
    """compute the cost by negative log likelihood."""
    return np.sum(np.log(1+np.exp(tx @ w))) - (y.T @ ( tx @ w))


def calculate_loss(y, tx, w, method="log"):
    """Calculate the loss.

    Calculate loss using MSE, MAE, RMSE or Neg Log Likelihood.
    (The last choice is our for this project)
    """
    if(method=="log"):
        return calculate_log(y, tx, w)
    elif(method=="rmse"):
        e = y - tx.dot(w)
        return calculate_rmse(e)
