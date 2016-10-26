import numpy as np

from helpers import *
from models import *
from evaluation import *
from gradient import *

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


#w = train(10000, 0.00001)
#predict(np.array(w))

y, x, ids = load_csv_data('data/train.csv')
tx = standardize(x)[0]
ws, losses, accus = cross_validation(y, tx, 5, 0, logistic_regression, 0.00001, 2000)
