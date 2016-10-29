# Implementation of the 6 ML methods
import numpy as np
from costs import calculate_loss
from gradient import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w, 'ls')
    return w, loss


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    batch_size = 100

    for n_iter in range(max_iters): 
        grad = compute_stoch_gradient(y, tx, w, batch_size)
        w = w - (gamma * grad)

    loss = calculate_loss(y, tx, w, 'ls')
    return w, loss


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
    loss = calculate_loss(y, tx, w, 'ls')
    
    return w, loss


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
    
    loss = calculate_loss(y, tx, w, 'ls')

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
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
    w = initial_w
    
    previous_loss = 0
    for n_iter in range(max_iters):
        # get loss and update w.
        loss = calculate_loss(y, tx, w)
        gradient = calculate_gradient_sigmoid(y, tx, w)
        w = w - gamma * gradient
        
        if n_iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}, gradient={g}"
                  .format(i=n_iter, l=loss, g=np.linalg.norm(gradient)))
        # converge criteria
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
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
    w = initial_w
    
    previous_loss = 0
    for n_iter in range(max_iters):
        # get loss and update w.
        penalty = lambda_ * (w.T @ w)
        loss = calculate_loss(y, tx, w) + penalty
        gradient = calculate_gradient_sigmoid(y, tx, w)
        w = w - gamma * gradient
        
        if n_iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}, gradient={g}"
                  .format(i=n_iter, l=loss, g=np.linalg.norm(gradient)))
        # converge criteria
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    return w, loss
