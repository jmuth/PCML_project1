import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from logistic_regression import *
from split_features import *


def build_poly(x, degree):
    """
    Build polynomial matrix for n-dimension data matrix x
    """
    rows = x.shape[0]
    n_features = x.shape[1]
    columns = (degree+1) * n_features

    #create the matrix tx
    tx = np.ones((rows, columns))

    for j in range(n_features):
        for k in range(degree+1):
            tx[:, j*(degree+1)+k] = x[:,j]**k

    return tx