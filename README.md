# Higgs Boson Machine Learning Challenge
Project 1 of PCML course, EPFL 2016.

## Group 17:

Joachim Muth, SCIPER 214757, joachim.muth@epfl.ch

Junze Bao, SCIPER 266983, junze.bao@epfl.ch

Chanhee Hwang, SCIPER 260872, chanhee.hwang@epfl.ch


## Environment
!IMPORTANT: We utilized the syntax of **"@"** when multiplying two matrices, and this syntax is only available in **Python3.5+**, so be sure to check the version of your local **/usr/local/bin/python**, which is the default runner if you run the script *run.py* directly by calling `./run.py`. Otherwise, explicitly specify python with the correct version, e.g. `python3 run.py`.


## Introduction

From data recorded in CERN Large Hadron Collider experiments, we will try to "find" Higgs boson particles. We have a classified dataset on which we will test mainly two types of algorithms: linear regression of  least squares or L2-regularized ridge regression, and logistic regression (with L2 regularizer).

We will analyze the provided data set in order to understand its features and to process them in a meaningful way. Following that, we will train our model with selected useful features, hoping to achieve a high accuracy in the prediction process.

With k-cross validation method, we will experiment locally with various parameters to find the best one, which achieves the highest accuracy, for each model. Afterwards, use the most precise model to predict the target variable of test dataset. In the end, our predictions will be submitted a [Kaggle competition](https://inclass.kaggle.com/c/epfml-project-1).

## Implementations of six model functions
All methods follow a similar procedure: first obtain the optimal *w* either by (stochastic) gradient descent or normal equations, and then calculate the loss with optimal *w*.

\# TODO: add more description!


## Other modules
Apart from the provided modules, we also have other 8 python module files to provide several functions:

1. *api.py*: It provides two basic public api for training and predicting. 7 arguments can be passed in to configure the training process. Prediction can also be adjusted with 7 options.

2. *gradient.py*: It provides 4 methods related to gradient descent, calculating (stochastic) gradient and hessian etc., and 1 method for calculating the sigmoid

3. *costs.py*: 4 methods to calculate different loss are provided. 1 method *calculate_loss()* can be used in general.

4. *split.py*: It provides one api to split the given data with specified granularity. Two specific methods can be used to split data into 4 or 8 groups, using the inner private *_split()* method.

5. *evaluation.py*: Cross validation can be conducted with the *cross_validation()* function provided in this module. It also gives us the accuracy, namely how many events we have correctly predicted, of each cross validation.

6. *helpers.py*: We added several functions to make this module richer than the provided one. *replace_num()* can be used to deal with missing values, i.e. replace missing value with median or mean value of others. *build_poly()* helps us to construct more complex polynomial feature matrix. We add one argument to *predict_labels()* to correlate the prediction method with training method.

7. *search_parameters.py*: It allows us to do cross validation to find best parameters for each group.

8. *method_comparison.py*: It allows us to compare the accuracy of each model and then choose the best one.
