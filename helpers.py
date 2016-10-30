# -*- coding: utf-8 -*-
"""some helper functions."""
import csv
import numpy as np
from gradient import *


def standardize(x, minX = None, rangeX = None):
    """Standardize the original data set.
    Using feature scaling:
    X = (X - Xmin) / (Xmax - Xmin)
    """
    if minX is None:
        minX = x.min(0)

    if rangeX is None:
        rangeX = x.max(0) - x.min(0)
        rangeX[rangeX==0] = 1

    x  = (x - minX) / rangeX
    return x, minX, rangeX


def replace_num(x, origin, target):
    """
    replace missing values

    Params:
        origin (int): one value in a feature column that's being replaced
        target (str): substitute value, e.g. mean, median
    """
    origin_ids = np.where(x == origin)
    x[origin_ids] = np.nan  # facilitate calculating mean
    if target == 'mean':
        mean_x = np.nanmean(x, axis=0)
        mean_x[np.isnan(mean_x)] = 0
        x[origin_ids] = np.take(mean_x, origin_ids[1])
    elif target == 'median':
        median_x = np.nanmedian(x, axis=0)
        median_x[np.isnan(median_x)] = 0
        x[origin_ids] = np.take(median_x, origin_ids[1])
    else:
        raise Exception('check target value')
    return x
    

def build_poly(x, degree):
    """
    polynomial basis functions for input data x, for j=0 up to j=degree.

    """
    #create the matrix tx
    poly_res = []
    for row_x in x:
        new_row = []
        for row_i in row_x:   # expand one row to three times its original size
            new_row += [row_i ** d for d in range(1, degree + 1)]
        poly_res.append(new_row)
    return np.array(poly_res)



def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def load_csv_data(data_path, sub_sample=False, background_value = 0):
    """
    Load data from csv files
    
    Params:
        data_path (str): file path and name
        sub_sample (boolean): if returned result is a sample
    
    Return:
        yb (ndarray): binary numeric value of target variable
        input_data (ndarray): feature matrix
        ids (ndarray): an array of ids of particles
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (0,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = background_value
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_header(data_path):
    """
    Load header of specified csv file
    
    Params:
        data_path (str): file path and name
        
    Return: 
        label (ndarray): csv headers
    """
    label = np.genfromtxt(data_path, delimiter=",", skip_header=0, max_rows=1, dtype= str)
    return label


def predict_labels(cut, weights, data, method='log'):
    """
    Generates class predictions given weights, and a test data matrix
    
    Params:
        weights (ndarray): optimal weight vector obtained by training
        data (ndarray): test matrix
    
    Return:
        y_pred (ndarray): an array of prediction results
    """
    if method == 'log':
        y_pred = sigmoid(data @ weights)
    else:
        y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred < cut)] = -1
    y_pred[np.where(y_pred >= cut)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    
    Params: 
        ids (ndarray):event ids associated with each prediction
        y_pred (ndarray): predicted class labels
        name (str): string name of .csv output file to be created
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
