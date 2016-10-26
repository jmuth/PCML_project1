import numpy as np

from helpers import *
from models import *
from evaluation import *
from gradient import *
from split import *

def train(n_iters, gamma):
    y, x, ids = load_csv_data('data/train.csv')
    tx = standardize(x)[0]
    loss, w = logistic_regression(y, tx, gamma, n_iters)
    print('in train, w is {}'.format(w))
    return w


def predict(w):
    """
    predict results are stored in res.csv
    """
    test_x = np.genfromtxt('data/test.csv', delimiter=',', skip_header=1)
    test_x = standardize(test_x[:, 2:])[0]
    create_csv_submission([i for i in range(350000,918238)], predict_labels(w, test_x), 'res.csv')

gamma = 0.00001
n_iters = 2000

# Normal train and predict 
#w = train(10000, 0.00001)
#predict(np.array(w))

# split train and predict
y, x, ids = load_csv_data('data/train.csv')
split_train = split_jets(y, x)
test_y, test_x, test_ids = load_csv_data('data/train.csv')
split_test = split_jets(test_y, test_x)

ws = []
for group in split_train:
    sub_y, sub_x, id_indices = group
    sub_tx = standardize(sub_x)[0]
    loss, w = logistic_regression(sub_y, sub_tx, gamma, n_iters)
    ws.append(w)

res = {}
for index, group in enumerate(split_test):
    sub_y, sub_x, id_indices = group
    sub_tx = standardize(sub_x)[0]
    pred_y = predict_labels(ws[index], sub_tx)
    res.update(dict(zip(test_ids[id_indices], pred_y)))

create_csv_submission(res.keys(), res.values(), 'split_res.csv')

# Cross Validation
# ws, losses, accus = cross_validation(y, tx, 5, 0, logistic_regression, 0.00001, 2000)

