import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, hp, STATUS_OK, Trials, tpe
import warnings
from FEATURES import AddFeatures

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
np.random.seed(42)

train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)

add_features = AddFeatures(train, test)
add_features.add_bmi_sist_dist_map()
add_features.add_f_score()

train = add_features.train
test = add_features.test

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train.as_matrix()


def hyperopt_train_test(hpparams):

    params_est = {
        'n_estimators': int(hpparams['n_estimators']),
        'criterion': 'gini',
        'max_features': hpparams['max_features'],
        'min_samples_split': hpparams['min_samples_split'],
        'min_samples_leaf': 1,
      }

    pipeline = RandomForestClassifier(**params_est)
    scores = model_selection.cross_val_score(pipeline, X, Y,
                                             cv=7, scoring='neg_log_loss', n_jobs=24)
    print(scores.mean())
    return scores.mean(), scores.std()


space4dt = {
   'n_estimators': hp.quniform('n_estimators', 1500, 5000, 1),
   'max_features': hp.quniform('max_features', 0.2, 0.8, 0.01),
   'min_samples_split': hp.quniform('min_samples_split', 0.01, 0.2, 0.01),
   # 'criterion': hp.choice('criterion', ('gini', 'entropy')),
}


def f(params):
    global log_, counter, params_, std_
    mlog, mstd = hyperopt_train_test(params)
    counter += 1

    log_.append(mlog)
    params_.append(params)
    std_.append(mstd)

    best_params = pd.DataFrame()
    best_params['log_loss'] = log_
    best_params['mstd'] = std_
    best_params['params'] = params_

    best_params.sort_values(by=['log_loss'], inplace=True, ascending=False)
    best_params.to_csv('search_RF_10_07_2.csv', index=False)

    return {'loss': abs(mlog), 'status': STATUS_OK, 'loss_variance': mstd}


trials = Trials()
log_, params_, std_ = [], [], []
counter = 0

best = fmin(f, space4dt, algo=tpe.suggest, max_evals=5000, trials=None, verbose=1)
print('best:')
print(best)
