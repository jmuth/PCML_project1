# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np


def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_rmse(y, tx, beta):
    """ compute the loss by rmse 
    """
    return np.sqrt(2 * compute_mse(y, tx, beta))