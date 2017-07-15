import pandas as pd
import numpy as np
import xgboost as xgb
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from FEATURES import AddFeatures
import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)
np.random.seed(42)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

add_features = AddFeatures(train, test)
add_features.add_bmi_sist_dist_map()
add_features.add_f_score()
add_features.add_ap_features()
# add_features.del_features()

train = add_features.train
test = add_features.test

Y = train['cardio']
train.drop(['cardio', 'id'], axis=1, inplace=True)
X = train

params_est_default = {
    'learning_rate': 0.02,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.6,
    'reg_lambda': 0.7,
    'gamma': 0.1,
    'eval_metric': 'logloss',
}

i = 0
for column in X.columns:
    print(i, column)
    i += 1

print(X[:3])

d_train = xgb.DMatrix(X, label=Y)
model = xgb.cv(params_est_default, d_train, 200, nfold=7, stratified=True)
print('[DEFAULT VALUE]: %s' % min(model['test-logloss-mean']))

params_est = {
    'learning_rate': 0.02,
    'max_depth': 5,
    'silent': 1,
    'subsample': 0.6,
    'reg_lambda': 0.7,
    'gamma': 0.1,
    'n_estimators': 200,
}

estimator = xgb.XGBClassifier(**params_est)
sfs1 = SFS(estimator, k_features=(1, len(X.columns)), forward=True, floating=False,
           verbose=2, scoring='log_loss', cv=7, n_jobs=1)
sfs1 = sfs1.fit(X.as_matrix(), Y)
print(sfs1.k_feature_idx_)
print(sfs1.k_score_)

#
# """ FEATURE generator """
# print('\nBefore transformation: ', X.shape)
# columns = [
#     'height', 'ap_hi', 'ap_lo', 'age_y', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log'
# ]
#
#
# for i1, col1 in enumerate(columns):
#     for i2, col2 in enumerate(columns):
#         if col1 == col2:
#             continue
#         X['%s_%s_1' % (col1, col2)] = X[col1] / (X[col2] + 1)
#         X['%s_%s_2' % (col1, col2)] = X[col1] * X[col2]
# print('\nAfter transformation: ', X.shape)
#
# i = 0
# for column in X.columns:
#     print(i, column)
#     i += 1
#
# params_est = {
#     'learning_rate': 0.02,
#     'max_depth': 5,
#     'silent': 1,
#     'subsample': 0.6,
#     'reg_lambda': 0.7,
#     'gamma': 0.1,
#     'n_estimators': 100,
# }
#
# estimator = xgb.XGBClassifier(**params_est)
# sfs1 = SFS(estimator, k_features=(1, 100), forward=True, floating=False,
#            verbose=2, scoring='log_loss', cv=3, n_jobs=-1)
# sfs1 = sfs1.fit(X.as_matrix(), Y)
# print(sfs1.subsets_)
# print(sfs1.k_feature_idx_)
# print(sfs1.k_score_)

""" FEATURE generator"""


"""
results with fast code:
 (1, 3, 7, 8, 10, 11, 12, 14, 15, 17, 19, 20, 23, 24) = -0.616370076628

 ['height', 'ap_hi', 'active', 'age_y', 'ch_1', 'ch_2', 'ch_3', 'gl_2', 'gl_3', 'sist_formula', 'map', 'F_score',
  'ap_h_delta', 'ap_l_delta']

results with normal code:
 (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23) -0.556838879887

 ['gender', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y', 'ap_diff', 'ch_1', 'ch_2', 'ch_3', 'gl_1',
  'gl_2', 'gl_3', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log', 'ap_/']

---------------------------------------------------------------
AMAZON AWS EC2:
    simple result (200 3_folds):
    0.5387133
    SBS columns:
        0.5384747 (0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21)

    SBS features:
        0.538606 (3, 5, 6, 7, 8, 133, 10, 12, 13, 15, 272, 19, 45, 46, 58, 61, 191, 193, 68, 216, 220, 98, 104, 107)

    NEW Features:
    -0.538449 (3, 4, 5, 6, 7, 8, 9, 10, 11, 19, 20, 22, 23)


"""