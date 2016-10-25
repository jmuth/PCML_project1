import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt
from logistic_regression import *
from helpers import *

def apply_right_model(tx, ids, w0_nomass, w0, w1_nomass, w1, w23_nomass, w23):
	"""
	Choose the right model to apply to each data, based on the features

	Params:
		tx (ndarray): data matrix
		ids (ndarray): array of ID's we want to predict
		w (ndarray): array of w matrices to apply to each data
	Return:
		ys (ndarray): array with ids and predict y
	"""
	pred = []

	for i, x in enumerate(tx):
			pred.append([ids[i], choose_right_w(x, w0_nomass, w0, w1_nomass, w1, w23_nomass, w23)])

	return np.array(pred)


def  pred(x, w):
	prob = sigmoid(x @ w)
	# prob = x @ w
	# print(prob)
	if prob >= 0.5:
		return 1
	else:
		return -1

def choose_right_w(x, w0_nomass, w0, w1_nomass, w1, w23_nomass, w23):
	if x[22] == 0:
		if x[0] == -999:
			return pred(x, w0_nomass)
		else:
			return pred(x, w0)
	elif x[22] == 1:
		if x[0] == -999:
			return pred(x, w1_nomass)
		else:
			return pred(x, w1)
	else:
		if x[0] == -999:
			return pred(x, w23_nomass)
		else:
			return pred(x, w23)


def subset_id(tx):	
	print(tx)
	jet0 = []
	jet0_nm = []
	jet1 = []
	jet1_nm = []
	jet23 = []
	jet23_nm = []

	for i, x in enumerate(tx):

		if x[22] == 0:
			if x[0] == -999.:
				jet0_nm.append(i)
			else:
				jet0.append(i)
		elif x[22] == 1:
			if x[0] == -999.:
				jet1_nm.append(i)
			else:
				jet1.append(i)
		else:
			if x[0] == -999.:
				jet23_nm.append(i)
			else:
				jet23.append(i)

	return np.array(jet0), np.array(jet0_nm), np.array(jet1), np.array(jet1_nm), np.array(jet23), np.array(jet23_nm)



def split_on_jets(y, tx):
	"""
	Split the data set into 3 cathegories based on the #jet feature: jet0, jet1 and jet2-3

	Params:
		y (nd.array): prediction array
		tx (nd.array): data matrix

	Return: 
	    jet0, ..., jet23
	    y0, ..., y23
	"""
	jet0 = []
	jet1 = []
	jet23 = []

	y0 = []
	y1 = []
	y23 = []

	for i, x in enumerate(tx):
		if x[22] == 0:
			jet0.append(x)
			y0.append(y[i])
		elif x[22] == 1:
			jet1.append(x)
			y1.append(y[i])
		else:
			jet23.append(x)
			y23.append(y[i])

	return np.array(jet0), np.array(jet1), np.array(jet23), np.array(y0), np.array(y1), np.array(y23)


def split_on_mass(y, tx):
	NaN_mass = []
	mass = []

	y_NaN_mass = []
	y_mass = []

	for i, x in enumerate(tx):
		if x[0] == -999.:
			NaN_mass.append(x)
			y_NaN_mass.append(y[i])
		else:
			mass.append(x)
			y_mass.append(y[i])

	return np.array(NaN_mass), np.array(mass), np.array(y_NaN_mass), np.array(y_mass)



def count_higgs(y):
	tot = y.shape[0]
	higgs = np.count_nonzero(y==1)
	print("# of Higgs: ", higgs, "over a total of :", tot, "---> ", float(higgs)/float(tot), "%")

def print_nan_summary(tx, DATA_TRAIN_PATH = 'data/train.csv'):
	"""
	Print a summary of each feature of tX matrix and the number of NaN's it 
	contains.

	Params:
		y (ndarray): array of prediction
		tx (ndarray): matrix
		DATA_TRAIN_PATH (string): path to the train data set to acces feature names

	Return:
		summary (ndarray): the array with number of meaningfull variable in each features
	"""
	labels = load_header(DATA_TRAIN_PATH)
	labels = labels[range(2, labels.size)] # do not load "ID" and "Prediction" label

	txt = tx.T

	summary = [["--Label--", "--NaN values--", "--defined values--"]]
	ret = []

	for i, f in enumerate(txt):
		summary.append([labels[i], np.count_nonzero(f == -999.0),np.count_nonzero(f != -999.0)])
		ret.append(np.count_nonzero(f != -999.0))

	pretty_print(summary)
	return np.array(ret)

def select_features_without_nan(xt):
	list_nan = []
	for f in xt.T:
		list_nan.append(np.count_nonzero(f == -999.0))

	list_nan = np.array(list_nan)
	indices = np.where(list_nan == 0)
	return (xt[:, indices[0]], indices[0])


def print_five_numbers_summary(tX, DATA_TRAIN_PATH = 'data/train.csv'):
	# label names
	labels = load_header(DATA_TRAIN_PATH)
	labels = labels[range(2, labels.size)] # do not load "ID" and "Prediction" label
	# maximum
	maximums = np.amax(tX, axis=0)
	# minimum
	minimums = np.amin(tX, axis=0)
	# lower quartile
	low_quartile = np.percentile(tX, 25, axis = 0)
	# median
	median = np.median(tX, axis = 0)
	# upper quartile
	high_quartile = np.percentile(tX, 75, axis = 0)

	# transpose matrix and add label
	table = np.array([labels, minimums, low_quartile, median, high_quartile, maximums])
	table = transpose(table)
	pretty_print(table)



def pretty_print(matrix):
	s = [[str(e) for e in row] for row in matrix]
	lens = [max(map(len, col)) for col in zip(*s)]
	fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
	table = [fmt.format(*row) for row in s]
	print('\n'.join(table))
    
def transpose(matrix):
	return zip(*matrix)

def boxplot_five_number(tx):
	fig = plt.figure(figsize=(15, 6))
	plt.boxplot(tx)
	plt.show

def isJet0_nomass(x):
	return (x[:,22] == 0) & (x[:,0] == -999.)

def isJet0(x):
	return (x[:,22] == 0) & (x[:,0] == -999.)

def isJet1_nomass(x):
	return (x[:,22] == 1) & (x[:,0] == -999.)

def isJet1(x):
	return (x[:,22] == 1) & (x[:,0] == -999.)

def isJet23_nomass(x):
	return ((x[:,22] == 2) | (x[:,22] == 3)) & (x[:,0] == -999.)

def isJet23(x):
	return ((x[:,22] == 2) | (x[:,22] == 3)) & (x[:,0] != -999.)

def apply_right_model(testset, \
                  selected_jet0_nomass, selected_jet0, selected_jet1_nomass, selected_jet1, \
                  selected_jet23_nomass, selected_jet23, \
                  min0_nomass, min0, min1_nomass, min1, min23_nomass, min23, \
                  range0_nomass, range0, range1_nomass, range1, range23_nomass, range23, \
                  w0_nomass, w0, w1_nomass, w1, w23_nomass, w23):
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



def apply_right_model_withoutmass_ridge(testset, \
                  selected_jet0, selected_jet1, \
                  selected_jet23, \
                  min0, min1, min23, \
                  range0, range1, range23, \
                  w0, w1, w23):
    y = []

    for x_t in testset:
        x = np.array([x_t])
        if isJet0_nomass(x) or isJet0(x):
            pred = x[:, selected_jet0]
            pred, _ , _  = standardize_mean(pred, min0, range0)
            y.append(predict(pred, w0))
        elif isJet1_nomass(x) or isJet1_nomass(x):
            pred = x[:, selected_jet1]
            pred, _ , _  = standardize_mean(pred, min1, range1)
            y.append(predict(pred, w1))
        else:
            pred= x[:, selected_jet23]
            pred, _ , _  = standardize_mean(pred, min23, range23)
            y.append(predict(pred, w23))


    y = np.array(y)
    
    return y





























