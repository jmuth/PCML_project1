from costs import *


def one_round_cross_validation(y, tx, k, k_indices, seed, model_func, *args, **kwargs):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k]]
   
    # find other indices 
    not_k = np.array([i for i in range(len(y)) if i not in k_indices[k]])
    y_train = y[not_k]
    tx_train = tx[not_k]

    # run model functions 
    loss_tr, w = model_func(y_train, tx_train, *args)
    loss_te = calculate_loss(y_test, tx_test, w, *args)
    func_name = 'LS' if model_func.__name__ == 'least_squares' else 'LR'
    accuracy = validation_accuracy(y_test, tx_test, w, func_name)
    return w, loss_tr, loss_te, accuracy 


def cross_validation(y, tx, k_fold, seed, model_func, *args, **kwargs):
    """
    Run cross validation on our dataset to see model performance
    
    Params:
        y (ndarray): target variable, usually a column vector
        tx (ndarray): independent variable matrix
        k (int): cross validation times, k fold
        seed (int): random seed
        model_func (func): model function
        *args (tuple): model function parameter tuple
        **kwargs (dict): model function parameter dict
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
    losses = []
    accuracies = []
    for ki in range(k_fold):
        # run CV k times, get accuracy each time
        # store w and loss for final evaluation
        w, loss_train, loss_test, accuracy = one_round_cross_validation(
                y, tx, ki, k_indices, seed, model_func, *args, **kwargs)
        ws.append(w)
        losses.append(loss_train)
        accuracies.append(accuracy)
    
    return ws, losses, accuracies


def validation_accuracy(y_test, tx_test, w, func_name):
    if func_name == 'LR':
        pred_y = predict_labels(w, tx_test)
    elif func_name == 'LS':
        pred_y = tx_test @ w
        pred_y[pred_y > 0] = 1
        pred_y[pred_y <= 0] = -1
    correct_count = 0
    for predict_y, true_y in zip(pred_y, y_test):
         correct_count += predict_y == true_y
    
    return correct_count / len(y_test)

def test_accuracy(y_test, y_predict):
    acc = (y_predict + y_test)
    return (1 - np.count_nonzero(acc == 0) / y_predict.shape[0])


