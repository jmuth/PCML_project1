import implementations
# least squares GD

def choose_best(cv, best):
    if cv[2] < best[2]:
        return cv
    else:
        return best
    
    
def cross_validate_parameters(y, x, function_name, method_name,cut=0.5, gammas=None, lambdas=None, *args, **kwargs):
    best = 0,0, float('inf'), 0
    
    if (lambdas is None) and (gammas is not None):
        for i in gammas:
            cv = api.train(y, x, poly=0, split_method=None, replace=None, cv=True, cut=cut, \
                           model_func=method_name, gamma = i, *args)
            best = choose_best(cv, best)
            
    elif (gammas is None) and (lambdas is not None):
        for i in lambdas:
            cv = api.train(y, x, poly=0, split_method=None, replace=None, cv=True, cut=cut, \
                           model_func=method_name, lambda_ = i, *args)
            best = choose_best(cv, best)
            
    elif (gammas is not None) and (lambdas is not None):
        for i in gammas:
            for j in lambdas:
                cv = api.train(y, x, poly=0, split_method=None, replace=None, cv=True, cut=cut, \
                               model_func=method_name, gamma = i, lambda_ = j, *args)
                best = choose_best(cv, best)
                
    else:
        best = api.train(y, x, poly=0, split_method=None, replace=None, cv=True, cut=cut, \
                               model_func=method_name, *args)
        
    print("-------------------------")
    print(function_name)
    print("te_loss = %.4f" % best[2])
    print("accuracy = %.4f" % best[3])

    return best


def run():
  # range parameter
  gammas = np.linspace(start=0.000001, stop=1, num=10)
  lambdas = np.logspace(start=-8, stop=10, num=10)
  max_iters = 10000


  # least squares
  cross_validate_parameters(y, x, 'Least Square', implementations.least_squares)

  # least squares GD
  cross_validate_parameters(y, x,'Least squares GD',  implementations.least_squares_GD, \
                                                  gammas = gammas, max_iters = max_iters)

  # least squares Stochastic GD
  cross_validate_parameters(y, x,'Least squares SGD', implementations.least_squares_SGD, \
                                                   gammas = gammas, max_iters = max_iters)

  # RR
  cross_validate_parameters(y, x, 'Ridge Regression', implementations.ridge_regression, lambdas = lambdas)


  # LR
  cross_validate_parameters(y_rescaled, x_rescaled,'Logistic Regression', implementations.logistic_regression, \
                                                    cut = 0.5, gammas = gammas, max_iters = max_iters)

  # RLR
  cross_validate_parameters(y_rescaled, x_rescaled, 'Regularized Logistic Regression', \
                            implementations.reg_logistic_regression, cut = 0.5, gammas = gammas, \
                            lambdas = lambdas, max_iters = max_iters)



if __name__ 












