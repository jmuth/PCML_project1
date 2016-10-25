# Implementation of the 6 ML methods
import numpy as np
from costs import *
from gradient import *


def least_squares_GD(y, tx, gamma, max_iters):
    """
    Linear regression using gradient descent
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        gamma (int): step size of gradient descent
        max_iters (int): iteration numbers
        
    Return:
        loss (float): final loss after max_iterations
        w (ndarray): optimal weight vector
    """
    w = np.zeros(tx.shape[1])
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w)
    return loss, w


def least_squares_SGD(y, tx, gamma, max_iters):
    """
    Linear regression using stochastic gradient descent
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        gamma (int): step size of gradient descent
        max_iters (int): iteration numbers
        
    Return:
        loss (float): final loss after max_iterations
        w (ndarray): optimal weight vector
    """
    w = np.zeros(tx.shape[1])
    batch_size = 100

    for n_iter in range(max_iters): 
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w, method="rmse")
    return loss, w


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        
    Return:
        loss (float): final loss 
        w (ndarray): optimal weight vector
        
    Equation:
        (X.T @ X) @ w = X.T @ y
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = calculate_loss(y, tx, w)
    
    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        lambda_ (float): regularization paramter
        
    Return:
        loss (float): final loss
        w (ndarray): optimal weight vector
       
    Equation:
        (X.T @ X + lambda_prime @ Im) @ w = X.T @ y 
            |              |                    |
        first_part   second_part           third_part 
    """
    lambda_prime = lambda_ * 2 * len(y)
    
    first_part = tx.T @ tx
    second_part = lambda_prime * np.identity(tx.shape[1])   # 2N*lamb*Im
    third_part = tx.T @ y
    w = np.linalg.solve(first_part+second_part, third_part)
    
    loss = calculate_loss(y, tx, w)

    return loss, w

# def ridge_regression_GD(y, tx, lambda_, max_iter=10000):
#     lambda_prime = lambda_ * 2 * len(y)

#     first_part = tx.T @ tx
#     second_part = lambda_prime * np.identity(tx.shape[1])   # 2N*lamb*Im
#     third_part = tx.T @ y
#     optimal_w = np.linalg.solve(first_part+second_part, third_part)
    
#     loss = calculate_loss(y, tx, w)

#     return loss, w


def logistic_regression(y, tx, gamma, max_iter=1000):
    """
    Logistic regression using gradient descent 
    we don't use Newton's method because computing hessian
    takes too much time
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        gamma (int): step size of logistic regression GD
        max_iters (int): iteration numbers
        
    Return:
        loss (float): final loss after max_iterations
        w (ndarray): optimal weight vector
    """
    # init parameters
    threshold = 1e-8
    previous_loss = 0
    w = np.zeros(tx.shape[1])
    
    previous_loss = 0
    for n_iter in range(max_iter):
        # get loss and update w.
        loss = calculate_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        
        if n_iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gradient={g}"
                  .format(i=n_iter, l=loss, g=np.linalg.norm(gradient)))
        # converge criteria
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    print("Final loss={l}".format(l=loss))
    return loss, w


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """
    Regularized logistic regression using gradient descent or SGD
    Use L2 regularizer
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        gamma (int): step size of logistic regression GD
        max_iters (int): iteration numbers
        
    Return:
        loss (float): final loss after max_iterations
        w (ndarray): optimal weight vector
    """
    # init parameters
    threshold = 1e-8
    previous_loss = 0
    w = np.zeros(tx.shape[1])
    
    previous_loss = 0
    for n_iter in range(max_iter):
        # get loss and update w.
        penalty = lambda_ * (w.T @ w)
        loss = calculate_loss(y, tx, w) + penalty
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient
        
        if n_iter % 500 == 0:
            print("Current iteration={i}, the loss={l}, gradient={g}"
                  .format(i=n_iter, l=loss, g=np.linalg.norm(gradient)))
        # converge criteria
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    print("Final loss={l}".format(l=loss))
    return loss, w
