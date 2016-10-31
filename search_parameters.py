import api
import helpers
import implementations
import split

import numpy as np

def _inner_search(y, x, degrees, lambdas, optimize_on="te_loss"):
	if optimize_on == "te_loss":
		best = 0, 0, float("inf")
	elif optimize_on == "accu":
		best = 0, 0, -float("inf")

	for l in lambdas:
		for degree in degrees:
			_, _, te_loss, accu = api._inner_train(y, x, poly = degree, replace=None, cv=True, cut=0., model_func=implementations.ridge_regression, lambda_=l)

			if optimize_on == "te_loss":
				if te_loss < best[2]:
					best = (l, degree, te_loss)
			elif optimize_on == "accu":
				if accu > best[2]:
					best = (l, degree, accu)

	return best

def load_and_search(sample = True, optimize_on = "te_loss", split_method = 'mass', standardization = False):

	# load the data
	print("load train set")
	y, x, _ = helpers.load_csv_data('data/train.csv', sub_sample = sample, background_value = -1)

	if standardization == True:
		y, _, _ = helpers.standardize_by_mean(y)
		x, _, _ = helpers.standardize_by_mean(x)

	#search
	return search(y, x, optimize_on, split_method = split_method)


def search(y, x, optimize_on="te_loss", split_method = 'mass'):
	"""
	Do a two-step parameter step between degree (2 to 15) and lambda (e-10 to e5) for Ridge Regression 
	method using 4-fold cross validation
	Find best parameter and zoom around lambda to have more precision.

	Parameter:
		sample: proceed the search on sample or on complete dataset (warning: slow!)

	Return:
		lambdas_star: array of best lambda for 8 models
		degrees_star: array of best degree for 8 models
	"""

        
	# split the data (8 model)
	split_train = split.split(y, x, method= split_method)

	# large range of parameter
	degrees = range(2, 15)
	lambdas = np.logspace(-5, 10)


	lambdas_star = []
	degrees_star = []
	print("start search")
	for i, splitted_set in enumerate(split_train):
		sub_y, sub_x, id_indices = splitted_set

		# first rough search with large scale
		lambda_star, degree_star, score = _inner_search(sub_y, sub_x, degrees, lambdas, optimize_on)

		# zoomed search around best parameters
		# zoomed_degree = range(degree_star-2, degree_star + 2)
		zoomed_lambda = np.logspace(lambda_star - 2, lambda_star + 2, 25)
		lambda_star, degree_star, score = _inner_search(sub_y, sub_x, degrees, zoomed_lambda, optimize_on)

		# store found values
		lambdas_star.append(lambda_star)
		degrees_star.append(degree_star)

		# print summary
		print("-------------------------------------")
		print("Set", i)
		print("-------------------------------------")
		print("lambda*:", lambda_star)
		print("degree: ", degree_star)
		if optimize_on == "te_loss":
			print("test set loss: ", score)
		elif optimize_on == "accu":
			print("accuracy: ", score)

	print("...............................")
	print("end")
	return lambdas_star, degrees_star	




