import numpy as np
import pandas as pd
import warnings
from FOLD_checker import FoldChecker

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
np.random.seed(42)

""" CONFIGURATION """
train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)
search_results = pd.read_csv('search_results/search_XGB_13_07.csv')
N_CHECKS = 20
N_CPU = 3

""" Checker matrix """
boosting_rounds_matrix = list(search_results['rounds'].values[:N_CHECKS])
params_matrix = list(search_results['params'].values[:N_CHECKS])

""" Main """
checker = FoldChecker(train, test)
return_results = pd.DataFrame()
for iteration in range(N_CHECKS):
    print('[SERVER #%s]' % iteration)
    iter_params = eval(params_matrix[iteration])
    iter_params['eval_metrics'] = 'logloss'
    iter_params['silent'] = 1
    iter_params['nthread'] = N_CPU

    results = checker.get_scores_xgb(state_iter=iteration, rounds=boosting_rounds_matrix[iteration],
                                     params_est=iter_params)
    results['server'] = iteration
    return_results = return_results.append(results)
    return_results.to_csv('fold_checker_results.csv')






