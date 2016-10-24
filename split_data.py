import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt



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

	jet0, jet1, jet23, y0, y1, y23 = split_on_jets(tX)


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
		void: print the summary
	"""
	labels = load_header(DATA_TRAIN_PATH)
	labels = labels[range(2, labels.size)] # do not load "ID" and "Prediction" label

	txt = tx.T

	summary = [["--Label--", "--NaN values--", "--defined values--"]]

	for i, f in enumerate(txt):
		summary.append([labels[i], np.count_nonzero(f == -999.0),np.count_nonzero(f != -999.0)])

	pretty_print(summary)

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

