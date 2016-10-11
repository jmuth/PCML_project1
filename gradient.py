# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np


def compute_gradient(y, tx, w):
	'''
	Compute the gradient
	'''
    e = np.array(compute_e(y, tx, w))
    N = len(e)
    tx_transp = np.transpose(tx)
    grad = np.dot(tx_transp, e)
    grad= - grad / N
    
    return grad