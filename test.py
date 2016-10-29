import numpy as np

from helpers import *
from evaluation import *
from gradient import *
from split import *
from plots import *


def train(y, x, poly, split, replace, cv, cut, model_func, *args, **kwargs):
    if split:
        split_train = split_jets(y, x)
        ws = []
        for group in split_train:
            sub_y, sub_x, id_indices = group
            w = inner_train(sub_y, sub_x, poly, replace, cv, cut, model_func, *args, **kwargs)
            ws.append(w)
        return ws
    else:
        w = inner_train(y, x, poly, replace, cv, cut, model_func, *args, **kwargs)
        return w

    
def inner_train(y, x, poly, replace, cv, cut, model_func, *args, **kwargs):
    if replace:
        x = replace_num(x, -999, replace)
    if poly != 0:
        x = build_poly(x, poly)
    tx = standardize(x)[0]
    if cv:
        w, tr_loss, te_loss, accu = cross_validation(y, tx, 5, 0, cut, model_func, *args, **kwargs)
        return w, tr_loss, te_loss, accu
    else:
        w, _ = model_func(y, tx, *args, **dict(kwargs, initial_w=np.zeros(tx.shape[1])))
    return w


def predict(test_y, test_x, test_ids, cut, w, poly=3, split=False, replace='median', res_to_file=False, method='log'):
    """
    predict results are stored in res.csv
    """
    if split:
        split_test = split_jets(test_y, test_x)  # test_y is []
        res = {}
        for index, group in enumerate(split_test):
            _, sub_x, id_indices = group
            pred_y = inner_predict(cut, w[index], sub_x, poly, replace, method)
            res.update(dict(zip(test_ids[id_indices], pred_y)))
        if res_to_file:
            create_csv_submission(res.keys(), res.values(), 'predict_split.csv')
        return res
    else:
        pred_y = inner_predict(cut, w, test_x, poly, replace, method)
        if res_to_file:
            create_csv_submission(test_ids, pred_y, 'predict.csv')
        
        return pred_y


def inner_predict(cut, w, x, poly, replace, method):
    if replace:
        x = replace_num(x, -999, replace)
    if poly != 0:
        x = build_poly(x, poly)
    tx = standardize(x)[0]
    pred_y = predict_labels(cut, w, tx, method)
    return pred_y

