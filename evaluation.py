from costs import *
from gradient import *
from helpers import *


def one_round_cross_validation(y, tx, k, k_indices, seed, cut, model_func, *args, **kwargs):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k]]

    # find other indices
    not_k = [i for i in range(len(y)) if i not in k_indices[k]]
    y_train = y[not_k]
    tx_train = tx[not_k]

    # run model functions
    if 'initial_w' in kwargs:
        kwargs = dict(kwargs, initial_w=np.zeros(tx_train.shape[1]))
    w, loss_tr = model_func(y_train, tx_train, *args, **kwargs)
    method = 'log' if 'logistic_regression' in model_func.__name__ else 'ls'
    loss_te = calculate_loss(y_test, tx_test, w, method)
    accuracy = validation_accuracy(y_test, tx_test, w, cut, method)
    #print('{} round, train loss {}, test loss {}, accuracy {}'.format(k, loss_tr, loss_te, accuracy))
    return w, loss_tr, loss_te, accuracy 


def cross_validation(y, tx, k_fold, seed, cut, model_func, *args, **kwargs):
    """
    Run cross validation on our dataset to see model performance
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        k (int): cross validation times, k fold
        seed (int): random seed
        model_func (func): model function
        *args (tuple): model function parameter tuple
        CAVEAT: **method** can be used to indicate the type of loss function, e.g. log/rmse 

    Return:
        avg_loss (float): average loss after k fold
        w (ndarray): average w after k fold
    """
    # build k_indices
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    ws = []
    tr_losses = []
    te_losses = []
    accuracies = []
    for ki in range(k_fold):
        # run CV k times, get accuracy each time
        # store w and loss for final evaluation

        w, loss_train, loss_test, accuracy = one_round_cross_validation(
                        y, tx, ki, k_indices, seed, cut, model_func, *args, **kwargs)
        ws.append(w)
        tr_losses.append(loss_train)
        te_losses.append(loss_test)
        accuracies.append(accuracy)
    
    return np.mean(ws), np.mean(tr_losses), np.mean(te_losses), np.mean(accuracies)


def validation_accuracy(y_test, tx_test, w, cut, method):
    """
    calculate the accuracy of the obtained parameter w

    Params:
        y_test (1-D ndarray): true y value
        tx_test (2-D ndarray): feature matrix
        w (1-D ndarray): optimal w obtained by training
        cut (float): patition value when predicting
        method (str): loss function name, i.e. log or ls
    """
    pred_y = predict_labels(cut, w, tx_test, method)
    correct_count = 0
    for predict_y, true_y in zip(pred_y, y_test):
        # pred_y belongs to {-1, 1}
        # y_test belongs to {0, 1}
        if predict_y == -1 and true_y == 0 or \
            predict_y == true_y:
            correct_count += 1
    return correct_count / len(y_test)


