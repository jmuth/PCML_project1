# -*- coding: utf-8 -*-
"""A function to compute the cost."""

import numpy as np
from helpers import *


def compute_gradient(y, tx, w):
    """
    Compute the gradient
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        w (ndarray): parameter matrix
    
    Return:
        grad (ndarray): gradient for multiple linear regression,
        under MS

    Equation:
        dL = - 1 / N * X.T @ (Y - X @ w)
    """
    e = y - tx.dot(w)
    n = len(e)
    grad = -(tx.T @ e) / n
    return grad


def compute_stoch_gradient(y, tx, w, batch_size):
    """
    Compute a stochastic gradient for batch data.
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        w (ndarray): parameter matrix
        batch_size (int): size of one batch
    
    Return:
        grad (ndarray): gradient for multiple linear regression,
        under MSE
    """
    stoch_grad = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        stoch_grad += compute_gradient(minibatch_y, minibatch_tx, w)
        
    stoch_grad = stoch_grad / batch_size
    return stoch_grad


def calculate_gradient_sigmoid(y, tx, w):
    """
    compute the gradient of loss with sigmoid function.
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        w (ndarray): parameter matrix
    
    Return:
        grad (ndarray): gradient for multiple linear regression,
        under MSE
        
    Equation:
        dL = X.T @ (sigma(X @ w) - Y)
    """
    return tx.T @ (sigmoid(tx @ w) - y)



def sigmoid(t):
    """apply sigmoid function on t.
    
    Params:
        t: t could be a number in the case of x[i] @ w,
           or a vector when called by X @ w
    
    Return:
        sigmoid result of t, a number or a vector
    
    Equation:
        sigmoid(t) = exp(t) / (1 + exp(t))

    """
    exp_t = np.exp(t)
    res = exp_t / (1 + exp_t)
    # if t is a number and exp_t overflows,
    # expt_t / (1 + exp_t) approximates to 1
    if isinstance(exp_t, int):
        if np.any(np.isinf(exp_t)): 
            print('int overflow')
            return 1    # inf / (1 + inf)
    else:  # i is a vector
        # after the division, inf turns to nan
        if any(np.isnan(res)):
             res[np.isnan(res)] = 1
    return res


def calculate_hessian(y, tx, w):
    """
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        w (ndarray): parameter matrix
    
    Return:
        hession (ndarray): consists of second derivates
    
    Equation:
        H = X.T @ S @ X
        S is a diagnoal matrix with diagnoal entries being 
        sigmoid(Xn.T @ w)[1 - sigmoid(Xn.T @ w)]
    """
    sigmo = sigmoid(tx @ w)
    one_vec = np.ones(len(sigmo))
    minus_sigmo = one_vec - sigmo
    S_array = np.multiply(sigmo, minus_sigmo)
    S = np.diag(S_array)

    return tx.T @ S @ tx

