import numpy as np

from collections import OrderedDict

from helpers import *
from evaluation import *
from gradient import *
from split import *
from plots import *


def train(y, x, poly, split_method, replace, cv, cut, model_func, *args, **kwargs):
    """
    serves as public api function

    Params:
        y (ndarray): target variable vector
        x (ndarray): feature matrix
        poly (int): degree of polynomial functions; don't use poly function when it's None
        split_method (str): split based on 'jet' or 'mass'; don't split when it's None
        replace (str): the method used to calculate the substitute for missing value; 
                       don't replace when it's None
        cv (boolean): Cross validation
        cut (float): paritition value when predicting y
        model_func (str): should be a callable function. indicates which model is used
        *args, **kwargs, arguments of the model_func

    Return:
        when split is not None, return a list of w of each groups; otherwise, just one 
        w. when cv is True, the aforementioned return value contains not only w, but 
        train loss, test loss and accuracy of each round of cross validation.
    """
    if split_method:
        split_train = split(y, x, split_method)
        ws = []
        for group in split_train:
            sub_y, sub_x, id_indices = group
            w = _inner_train(sub_y, sub_x, poly, replace, cv, cut, model_func, *args, **kwargs)
            ws.append(w)
        return ws
    else:
        w = _inner_train(y, x, poly, replace, cv, cut, model_func, *args, **kwargs)
        return w

    
def _inner_train(y, x, poly, replace, cv, cut, model_func, *args, **kwargs):
    """
    private inner train function

    Params and Return
        same as public train function
    """
    if replace:
        x = replace_num(x, -999, replace)
    if poly != 0:
        x = build_poly(x, poly)
    tx = standardize(x)[0]
    if cv:
        w, tr_loss, te_loss, accu = cross_validation(y, tx, 5, 0, cut, model_func, *args, **kwargs)
        return w, tr_loss, te_loss, accu
    else:
        if 'initial_w' in kwargs:
            kwargs = dict(kwargs, initial_w=np.zeros(tx.shape[1]))
        w, _ = model_func(y, tx, *args, **kwargs)
    return w


def predict(test_y, test_x, test_ids, cut, w, poly, split_method, replace, loss_method='log', res_to_file=True):
    """
    this function serves as a public API for predicting 

    Params:
        test_y (ndarray): target variable
        test_x (ndarray): feature matrix of test data
        test_ids (ndarray): ids of each event
        cut (float): patition value when predicting y
        w (ndarray): obtained by training
        poly (int): degree of polynomial functions; don't poly when it's None
        split_method (str): split based on 'jet' or 'mass'; don't split when it's None
        replace (str): the method used to calculate the substitute for missing value; 
                       don't replace when it's None
        loss_method (str): specify which loss function to use
        res_to_file (boolean): whether save the prediction results to file

    Return:
        a list or a dict containing the prediction result for each event
    """
    if split_method:
        split_test = split(test_y, test_x, split_method)  # test_y is []
        res = {}
        for index, group in enumerate(split_test):
            _, sub_x, id_indices = group
            pred_y = _inner_predict(cut, w[index], sub_x, poly, replace, loss_method)
            res.update(dict(zip(test_ids[id_indices], pred_y)))
        # sort res
        res = OrderedDict(sorted(res.items()))
        if res_to_file:
            create_csv_submission(res.keys(), res.values(), 'predict_split.csv')
        return res
    else:
        pred_y = _inner_predict(cut, w, test_x, poly, replace, loss_method)
        if res_to_file:
            create_csv_submission(test_ids, pred_y, 'predict.csv')
        
        return pred_y


def _inner_predict(cut, w, x, poly, replace, loss_method):
    """
    private inner predict function

    Params and return:
        same as public predict function
    """
    if replace:
        x = replace_num(x, -999, replace)
    if poly != 0:
        x = build_poly(x, poly)
    tx = standardize(x)[0]
    pred_y = predict_labels(cut, w, tx, loss_method)
    return pred_y

