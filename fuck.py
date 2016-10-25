import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from logistic_regression import *
from split_features import *

def apply_right_model(testset, \
                  selected_jet0_nomass, selected_jet0, selected_jet1_nomass, selected_jet1, \
                  selected_jet23_nomass, selected_jet23 \
                  min0_nomass, min0, min1_nomass, min1, min23_nomass, min23, \
                  range0_nomass, range0, range1_nomass, range1, range23_nomass, range23\
                 )
    y = []

    for x_t in testset:
        x = np.array([x_t])
        if isJet0_nomass(x):
            pred = x[:,selected_jet0_nomass]
            pred, _ , _ = standardize(pred, min0_nomass, range0_nomass)
            y.append(log_reg_predict(pred, w0_nomass))
        elif isJet0(x):
            pred = x[:, selected_jet0]
            pred, _ , _  = standardize(pred, min0, range0)
            y.append(log_reg_predict(pred, w0))
        elif isJet1_nomass(x):
            pred = x[:, selected_jet1_nomass]
            pred, _ , _  = standardize(pred,min1_nomass, range1_nomass)
            y.append(log_reg_predict(pred, w1_nomass))
        elif isJet1(x):
            pred = x[:, selected_jet1]
            pred, _ , _  = standardize(pred, min1, range1)
            y.append(log_reg_predict(pred, w1))
        elif isJet23_nomass(x):
            pred= x[:, selected_jet23_nomass]
            pred, _ , _  = standardize(pred, min23_nomass, range23_nomass)
            y.append(log_reg_predict(pred, w23_nomass))
        else:
            pred= x[:, selected_jet23]
            pred, _ , _  = standardize(pred, min23, range23)
            y.append(log_reg_predict(pred, w23))

    y = np.array(y)[:,0]
    
    return y








