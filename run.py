#!/usr/local/bin/python
import helpers    # you need to use load_csv_data
import api    # use train/predict method below
import numpy as np
import implementations
import helpers

standardization = True


def run():
	print("START")

	# load full dataset
	print("loading train set...")
	y_full, x_full, _ = helpers.load_csv_data('data/train.csv', sub_sample = False, background_value = -1)

	if standardization == True:
		y_full, _, _ = helpers.standardize_by_mean(y_full)
		x_full, _, _ = helpers.standardize_by_mean(x_full)
	# train the model with chosen parameters

	#LAMBDAS = lambdas_star
	#DEGREES = degrees_star

	#LAMBDAS = [1.59985871961e-05, 3.23745754282e-05, 0.152641796718, 3.23745754282e-05, 0.0184206996933, 7.90604321091e-06, 0.625055192527, 0.00910298177992]
	#DEGREES = [2, 2, 13, 2, 2, 3, 3, 3]

	LAMBDAS = [0.010000230261160271, 0.010000465959591297, 0.15598030775604094, 0.010000942933491358, 0.022731748970850542, 0.010000230261160271, 0.61745359148527057, 0.010268664246200506]
	DEGREES = [5, 6, 13, 14, 2, 3, 3, 3]


	# check accuracy
	print("checking accuracy...")
	w_cv = api.train(y_full, x_full, poly=DEGREES, split_method='mass', replace=None, cv=True, cut=0., \
		model_func=implementations.ridge_regression, lambdas = LAMBDAS)
	print("--------------------------------------------")
	print("Kaggle summary")
	print("  method: mass, replace: None, method: RR")
	print("  lambdas: ", LAMBDAS)
	print("  degrees: ", DEGREES)
	print("  accuracy mean :" ,np.mean(np.array(w_cv)[:,3]))
	print("  details: ", np.array(w_cv)[:,3])
	print("--------------------------------------------")

	# train de model
	print("training model...")
	w = api.train(y_full, x_full, poly=DEGREES, split_method='mass', replace=None, cv=False, cut=0., \
		model_func=implementations.ridge_regression, lambdas = LAMBDAS)

	# load test set
	print("loading test set...")
	test_y, test_x, test_ids = helpers.load_csv_data('data/test.csv')

	if standardization == True:
		test_y, _, _ = helpers.standardize_by_mean(test_y)
		test_x, _, _ = helpers.standardize_by_mean(test_x)

	# do the final prediction
	print("predicting...")
	final_pred = api.predict(test_y, test_x, test_ids, 0., w, poly=DEGREES, split_method='mass', \
		replace=None, loss_method='ls', res_to_file=True)

	print("Prediction produced in file predict_split.csv")
	print("FINISH")



if __name__ == '__main__':
	run()
