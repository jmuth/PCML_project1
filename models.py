# Implementation of the 6 ML methods
import numpy as np
from costs import calculate_loss
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
    # TODO: we have to find a good way to choose initial weight, maybe something random...
    # initial weights
    _, w = least_squares(y, tx)
    # w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w)
    return loss, w


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
    w = np.zeros(tx.shape[1])
    # TODO: choose a batch size
    batch_size = 100

    for n_iter in range(max_iters): 
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w, method="rmse")
    return loss, w


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
    loss = calculate_loss(y, tx, w)

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

    try:
        inversed = np.linalg.inv(bxx)
    except:
        raise ValueError("Matrix X^TX not invertible")

    w = inversed @ tx.T @ y
    loss = calculate_loss(y, tx, w)

    return loss, w


def logistic_regression(y, tx, gamma = 0.01,max_iters = 10000):
    """
    Logistic regression using gradient descent (Newton Method)
    :param y:
    :param tx:
    :param gamma:
    :param max_iters:
    :return:
    """
    # init parameters
    threshold = 1e-8
    losses = []
    # build w
    w = np.zeros((tx.shape[1], 1))
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = one_step_logistic_regression(y, tx, w, alpha)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return loss, w

def one_step_logistic_regression(y, tx ,w , gamma):
    """ One step og logistic regression using Newton method
    :param y:
    :param tx:
    :param gamma:
    :return: loss and weights
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient_sigmoid(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    w = w - alpha * np.linalg.inv(hessian) @ grad

    return loss, w


def reg_logistic_regression(y, tx, lambda_ = 0.01, gamma = 0.01, max_iters = 10000):
    """
    Regularized logistic regression using gradient descent or SGD
    :param y:
    :param tx:
    :param lambda_: regularisation parameter
    :param gamma: step size of gradient descent
    :param max_iters:
    :return: loss, weights
    """

    # init parameters
    threshold = 1e-8
    losses = []

    # build w
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = one_step_reg_logistic_regression(y, tx, w, lambda_, gamma)
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    print("The loss={l}".format(l=calculate_loss(y, tx, w)))
    return loss, w

def one_step_reg_logistic_regression(y, tx, w, lambda_, gamma):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    :param y:
    :param tx:
    :param lambda_: regularisation parameter
    :param gamma: step size of gradient descent
    :return: loss, weights
    """
    penalty = lambda_ * (w.T @ w)
    
    loss = calculate_loss(y, tx, w) + penalty
    grad = calculate_gradient_sigmoid(y, tx, w)
    hessian = calculate_hessian(y, tx, w)

    # print("debug step")
    # print("w before", w.shape)
    # print("hessian", hessian.shape)
    # print("grad", grad.shape)
    # print("gamma", gamma)
    w = w - gamma * np.linalg.inv(hessian) @ grad
    # print("w after", w.shape)
    return loss, w













