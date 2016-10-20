# Implementation of the 6 ML methods
import numpy as np
from costs import *
from gradient import *

def least_squares_GD(y, tx, gamma, max_iters):
    """
    Linear regression using gradient descent
    """
    # initial weights
    w = np.zeros(tx.shape[1])

    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)

    loss = compute_loss(y, tx, w)
    return loss, w


def least_squares_SGD(y, tx, gamma, max_iters):
    w = np.zeros(tx.shape[1])
    # TODO: choose a batch size
    batch_size = 100

    for n_iter in range(max_iters): 
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - (gamma * grad)

    loss = compute_loss(y, tx, w)
    return loss, w


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    :param y:
    :param tx:
    :return:
    """
    w = np.linalg.solve(tx.T @ tx, tx @ y)
    loss = compute_loss(y, tx, w)

    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    lambda_prime = lambda_ * 2 * len(y)
    xx = tx.T @ tx 
    bxx = xx + lambda_prime * np.identity(len(xx))

    w = np.linalg.solve(bxx, tx.T @ y) 
    loss = compute_loss(y, tx, w)

    return loss, w


def logistic_regression(y, tx, gamma,max_iters):
    """
    Logistic regression using gradient descent or SGD
    :param y:
    :param tx:
    :param gamma:
    :param max_iters:
    :return:
    """
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """
    Regularized logistic regression using gradient descent or SGD
    :param y:
    :param tx:
    :param lambda_:
    :param gamma:
    :param max_iters:
    :return:
    """
    raise NotImplementedError

