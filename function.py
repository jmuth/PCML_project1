# Implementation of the 6 ML methods
import numpy as np
from costs import *
from gradient import *


def least_squares_GD(y, tx, gamma, max_iters):
    """
    Linear regression using gradient descent
    :param y:
    :param tx:
    :param gamma:
    :param max_iters:
    :return:
    """
    # choose initial weights
    # TODO: we have to find a good way to choose initial weight, maybe something random...
    # for the moment, just ones
    # Define parameters to store w and loss
    initial_w = np.zeros(tx.shape[1])
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        # update w by gradient
        w = w - (gamma * grad)

        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)

    return losses, ws


def least_squares_SGD(y, tx, gamma, max_iters):
    """
    Linear regression using stochastic gradient descent
    :param y:
    :param tx:
    :param gamma:
    :param max_iters:
    :return:
    """
    # TODO: we have to find a good way to choose initial weight, maybe something random...
    initial_w = np.zeros(tx.shape[1])
    # TODO: choose a batch size
    batch_size = 100

    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        loss = compute_mse(y, tx, w)
        w = w - (gamma * grad)
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    :param y:
    :param tx:
    :return:
    """
    xx = np.dot(np.matrix.transpose(tx), tx)

    # handle non-inversable matrix case
    try:
        inv_xx = np.linalg.inv(xx)
    except:
        raise ValueError('Matrix X^TX is not invertible')

    w = np.matrix.dot(np.matrix.dot(inv_xx, np.matrix.transpose(tx)), y)
    loss = compute_mse(y, tx, w)

    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    :param y:
    :param tx:
    :param lambda_:
    :return:
    """
    xx = np.dot(np.transpose(tx), tx)

    bxx = xx + lambda_ * np.identity(len(xx))

    try:
        inv = np.linalg.inv(bxx)
    except:
        raise ValueError("Matrix X^TX not invertible")

    xy = np.dot(np.transpose(tx), y)
    w_star = np.dot(inv, xy)

    return w_star


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

