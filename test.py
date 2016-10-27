import numpy as np

from helpers import *
from models import *
from evaluation import *
from gradient import *
from split import *


def inner_train(y, x, gamma, n_iters, replace, cv):
    if replace:
        x = replace_num(x, -999, 'mean')
    tx = standardize(x)[0]
    if cv:
        cross_validation(y, tx, 5, 0, logistic_regression,
                gamma, n_iters)
        return 
    else:
        loss, w = logistic_regression(y, tx, gamma, n_iters)
    return w

def train(gamma, n_iters, poly=3, split=False, replace=False, cv=False):
    y, x, ids = load_csv_data('data/train.csv')
    if poly != 0:
        x = build_poly(x, poly)
    if split:
        split_train = split_jets(y, x)
        ws = []
        for group in split_train:
            sub_y, sub_x, id_indices = group
            w = inner_train(sub_y, sub_x, gamma, n_iters, replace, cv)
            ws.append(w)
        return ws
    else:
        w = inner_train(y, x, gamma, n_iters, replace, cv)
        return w


def predict(w, poly=3, split=False, res_to_file=False):
    """
    predict results are stored in res.csv
    """
    test_y, test_x, test_ids = load_csv_data('data/test.csv')
    if poly != 0:
        test_x = build_poly(test_x, poly)
    if split:
        split_test = split_jets(test_y, test_x)  # test_y is []
        res = {}
        for index, group in enumerate(split_test):
            _, sub_x, id_indices = group
            sub_tx = standardize(sub_x)[0]
            pred_y = predict_labels(0.5, w[index], sub_tx)
            res.update(dict(zip(test_ids[id_indices], pred_y)))
        if res_to_file:
            create_csv_submission(res.keys(), res.values(), 'predict_split.csv')
        else:
            return res
    else:
        test_tx = standardize(test_x)[0]
        pred_y = predict_labels(0.5, w, test_tx)
        if res_to_file:
            create_csv_submission(test_ids, pred_y, 'predict.csv')
        else:
            return pred_y
    

#gamma = 0.00001
#n_iters = 20000
#
#w = train(gamma, n_iters, split=True, replace=True)
#pred_y = predict(w, split=True, res_to_file=True)


# split train and predict
#y, x, ids = load_csv_data('data/train.csv')
#split_train = split_jets(y, x)
#test_y, test_x, test_ids = load_csv_data('data/test.csv')
#split_test = split_jets(test_y, test_x)
#
#ws = []
#for group in split_train:
#    sub_y, sub_x, id_indices = group
#    sub_tx = standardize(sub_x)[0]
#    cross_validation(sub_y, sub_tx, 5, 0, logistic_regression, gamma, n_iters)
#    #loss, w = logistic_regression(sub_y, sub_tx, gamma, n_iters)
#    #ws.append(w)
#
#res = {}
#for index, group in enumerate(split_test):
#    sub_y, sub_x, id_indices = group
#    sub_tx = standardize(sub_x)[0]
#    pred_y = predict_labels(0.5, ws[index], sub_tx)
#    res.update(dict(zip(test_ids[id_indices], pred_y)))
#
#create_csv_submission(res.keys(), res.values(), 'split_res.csv')
#
# Cross Validation
# ws, losses, accus = cross_validation(y, tx, 5, 0, logistic_regression, 0.00001, 2000)

