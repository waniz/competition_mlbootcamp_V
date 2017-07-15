import pandas as pd
import numpy as np
import xgboost as xgb
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import warnings
from FEATURES import AddFeatures

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
np.random.seed(42)

train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)

add_features = AddFeatures(train, test)
add_features.add_bmi_sist_dist_map()
add_features.add_f_score()
add_features.add_ap_features()
add_features.del_features()

train = add_features.train
test = add_features.test

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train

# best_columns = [
#     'gender', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y', 'ap_diff', 'ch_1', 'ch_2', 'ch_3', 'gl_1',
#     'gl_2', 'gl_3', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log', 'ap_/'
# ]


def hyperopt_train_test(hpparams):

    params_est = {
        'learning_rate': hpparams['eta'],
        'max_depth': hpparams['max_depth'],
        'gamma': hpparams['gamma'],
        'reg_lambda': hpparams['reg_lambda'],
        'min_child_weight': hpparams['min_child_weight'],
        'nthread': 20,
        'subsample': hpparams['subsample'],
        'colsample_bytree': hpparams['colsample_bytree'],
        'eval_metric': ['logloss', 'auc'],
        'silent': 1,
      }

    d_train = xgb.DMatrix(X, label=Y)
    model = xgb.cv(params_est, d_train, 4000, nfold=7, stratified=True, early_stopping_rounds=200)
    # print(model)
    print(round(min(model['test-logloss-mean']), 5),
          round(np.std(model['test-logloss-mean']), 5),
          round(max(model['test-auc-mean']), 5),
          len(model))
    return min(model['test-logloss-mean']), np.std(model['test-logloss-mean']), len(model['test-logloss-mean']), max(model['test-auc-mean'])


space4dt = {
   'max_depth': hp.choice('max_depth', (4, 5, 6, 7)),
   'eta': hp.quniform('eta', 0.00001, 0.01, 0.0001),
   'gamma': hp.quniform('gamma', 0, 0.16, 0.001),
   'reg_lambda': hp.quniform('reg_lambda', 0.7, 0.9, 0.01),
   'min_child_weight': hp.quniform('min_child_weight', 10, 100, 0.1),
   'subsample': hp.quniform('subsample', 0.3, 0.8, 0.01),
   'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1, 0.01),
}


def f(params):
    global log_, counter, params_, std_, rounds, auc
    mlog, mstd, round_, auc_ = hyperopt_train_test(params)
    counter += 1

    log_.append(mlog)
    params_.append(params)
    std_.append(mstd)
    rounds.append(round_)
    auc.append(auc_)

    best_params = pd.DataFrame()
    best_params['log_loss'] = log_
    best_params['auc'] = auc
    best_params['mstd'] = std_
    best_params['rounds'] = rounds
    best_params['params'] = params_

    best_params.sort_values(by=['log_loss'], inplace=True, ascending=True)
    best_params.to_csv('search_XGB_10_07.csv', index=False)

    return {'loss': mlog, 'status': STATUS_OK, 'loss_variance': mstd}


trials = Trials()
log_, params_, std_, rounds, auc = [], [], [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=500, trials=None, verbose=1)
print('best:')
print(best)
