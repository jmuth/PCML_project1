import helpers    # you need to use load_csv_data
import api    # use train/predict method below
import numpy as np
import implementations


def run():

	n_iters = 50000
	gamma = 0.000001
	lambda_ = 0.01
	initial_w = np.zeros(90)
	cut = 0.43
	poly = 2

	## Load data: train set
	print("Load data: train set")
	y, x, ids = helpers.load_csv_data('data/train.csv', backgroud_value = 0)

	## train model
	print("Train model")
	w = api.train(y, x, poly=poly, split_method='mass', replace=None, cv=False, cut=cut, 
	               model_func=implementations.reg_logistic_regression, 
	               lambda_=lambda_, initial_w=initial_w, max_iters=n_iters, gamma=gamma)

	## Load data: test set
	print("Load data: test set")
	test_y, test_x, test_ids = helpers.load_csv_data('data/test.csv')

	## Predict
	print("Predict y's")
	api.predict(test_y, test_x, test_ids, 0.43, w, poly=poly, split_method='mass', \
	 replace=None, res_to_file=True)





if __name__ == '__main__':
	run()

