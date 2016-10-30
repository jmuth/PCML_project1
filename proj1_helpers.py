# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
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
    yb[np.where(y=='b')] = 0
    
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


def predict_labels(weights, data):
    """
    Generates class predictions given weights, and a test data matrix
    
    Params:
        weights (ndarray): optimal weight vector obtained by training
        data (ndarray): test matrix
    
    Return:
        y_pred (ndarray): an array of prediction results
    """
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred

def predict(x, w):
	
	pred = x @ w
	if pred <= 0:
		return -1
	else:
		return 1

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