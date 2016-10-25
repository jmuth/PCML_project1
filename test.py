import numpy as np

from helpers import *
from models import *
from evaluation import *
from gradient import *

def train(n_iters, gamma):
    y, x, ids = load_csv_data('data/train.csv')
    tx = standardize(x)
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


# w = train(10000, 0.00001)
#w = [0.51502834, -14.63601116, -10.09640999, 3.94182946, 0.41693964
#,11.50271621,-0.75562943, 2.48822092,-0.81959088, 3.57958442
#,-11.7461814,0.57607797, 0.17223875, 9.14219569,-0.16497434
#,-0.22286615,11.06851602,-0.13596407,-0.17577068, 2.4593194
#,-0.12675413,-4.27253972,-1.99307728, 0.07803497, 0.50102021
#, 0.19005594,-1.86616439, 0.04445148,-0.08403331,-3.83714448]
#predict(np.array(w))

y, x, ids = load_csv_data('data/train.csv')
tx = standardize(x)[0]
ws, losses, accus = cross_validation(y, tx, 5, 0, logistic_regression, 0.00001, 10000)
