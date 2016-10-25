import numpy as np


def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    res = []
    for i in range(tx.shape[0]):
        sigma_xw = sigmoid(tx[i] @ w)
        snn = sigma_xw * (1 - sigma_xw)
        res.append(snn[0])
    return tx.T @ np.diag(res) @ tx


def newton_method(y, x, initial_w, max_iters):
    w = initial_w
    for n_iter in range(max_iters):
        hes = hessian(x, w)
        gd = log_reg_gradient(y, x, w)
        w = w - np.linalg.solve(hes, (hes @ w) + gd)
    return w


def sigmoid(t):
    """apply sigmoid function on t."""
    exp_t = np.exp(t)
    res = exp_t / (1 + exp_t)
    # # if t is a number and exp_t overflows,
    # # expt_t / (1 + exp_t) approximates to 1
    # if isinstance(exp_t, int):
    #     if np.any(np.isinf(exp_t)): 
    #         print('int overflow')
    #         return 1    # inf / (1 + inf)
    # else:  # i is a vector
    #     # after the division, inf turns to nan
    #     res[np.isnan(res)] = 1
    return res 


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return tx.T @ (sigmoid(tx @ w) - y)


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    for x_i, y_i in zip(tx, y):
        s = x_i @ w
        l = np.log(1 + np.exp(s))
        if np.isinf(l):  # when l overflows, uses s to approximate
            l = s
        loss += (l - y_i * s)
    if loss < 0:
        print('negative loss: {} {} {} {}'.format(loss, l, s, y_i))
    return loss


def logistic_reg_gd(y, tx, w, max_iter, gamma, threshold):
    # start the logistic regression
    previous_loss = 0
    for n_iter in range(max_iter):
        # get loss and update w.
        loss = calculate_loss(y, tx, w, "rmse")
        gradient = calculate_gradient(y, tx, w)
        w = w - gamma * gradient
        if n_iter % 100 == 0:
            print("Current iteration={i}, the loss={l}, gradient={g}".format(i=n_iter, l=loss, g=np.linalg.norm(gradient)))
        # converge criteria
        if np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss
    return w


def log_reg_predict(tx, w):
    probs = []
    for x in tx:
        prob = sigmoid(x @ w)
        if prob >= 0.5:
            # print('11')
            probs.append(1)
        else:
            probs.append(-1)
    return np.array(probs)
