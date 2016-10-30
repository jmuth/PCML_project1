import helpers    # you need to use load_csv_data
import api    # use train/predict method below
import numpy as np
import implementations


def run():
	print("START")

	# load full dataset
	print("loading train set...")
	y_full, x_full, _ = helpers.load_csv_data('data/train.csv', sub_sample = False, background_value = -1)

	# train the model with chosen parameters

	#LAMBDAS = lambdas_star
	#DEGREES = degrees_star

	LAMBDAS = [1.59985871961e-05, 3.23745754282e-05, 0.152641796718, 3.23745754282e-05, 0.0184206996933, 7.90604321091e-06, 0.625055192527, 0.00910298177992] 
	DEGREES = [2, 2, 13, 2, 2, 3, 3, 3]


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
	print("training model")
	w = api.train(y_full, x_full, poly=DEGREES, split_method='mass', replace=None, cv=False, cut=0., \
		model_func=implementations.ridge_regression, lambdas = LAMBDAS)

	# load test set
	print("loading test set...")
	test_y, test_x, test_ids = helpers.load_csv_data('data/test.csv')

	# do the final prediction
	print("producing prediction file...")
	final_pred = api.predict(test_y, test_x, test_ids, 0., w, poly=DEGREES, split_method='mass', \
		replace=None, loss_method='ls', res_to_file=True)

	print("Prediction produced in file predict_split.csv")
	print("FINISH")



if __name__ == '__main__':
	run()

