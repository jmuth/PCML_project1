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


def compute_loss(y, tx, w):
    """Calculate the loss.

    Calculate loss using RMSE
    """
    e = y - tx.dot(w)
    # return calculate_mse(e)
    # return calculate_mae(e)
    return calculate_rmse(e)