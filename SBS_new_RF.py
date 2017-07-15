import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from FEATURES import AddFeatures

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)
np.random.seed(42)

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

add_features = AddFeatures(train, test)
add_features.add_bmi_sist_dist_map()
add_features.add_f_score()
add_features.add_ap_features()
add_features.del_features()

train = add_features.train
test = add_features.test

Y = train['cardio']
train.drop(['cardio', 'id'], axis=1, inplace=True)
X = train

params_rf = {
    'n_estimators': 50,
    'criterion': 'gini',
    'max_features': 0.4,
    'min_samples_split': 0.008,
    'min_samples_leaf': 2,
}
# model = RandomForestClassifier(**params_rf)
# scores = cross_val_score(model, X, Y, scoring='log_loss', cv=4, n_jobs=-1, verbose=1)
# print(np.mean(scores))

# sfs1 = SFS(model, k_features=(1, len(X.columns)), forward=True, floating=False,
#            verbose=2, scoring='log_loss', cv=4, n_jobs=-1)
# sfs1 = sfs1.fit(X.as_matrix(), Y)
# print('Results:')
# print(sfs1.k_feature_idx_)
# print(sfs1.k_score_)


print('\nBefore transformation: ', X.shape)
columns = [
    'height', 'ap_hi', 'ap_lo', 'age_y', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log'
]
for i1, col1 in enumerate(columns):
    for i2, col2 in enumerate(columns):
        if col1 == col2:
            continue
        # X['%s_%s_1' % (col1, col2)] = X[col1] / (X[col2] + 1)
        X['%s_%s_2' % (col1, col2)] = X[col1] * X[col2]
print('\nAfter transformation: ', X.shape)

i = 0
for column in X.columns:
    print(i, column)
    i += 1

sfs1 = SFS(model, k_features=(10, 60), forward=True, floating=False,
           verbose=2, scoring='log_loss', cv=4, n_jobs=-1)
sfs1 = sfs1.fit(X.as_matrix(), Y)
print('Results:')
print(sfs1.k_feature_idx_)
print(sfs1.k_score_)


"""
  DEFAULT:     0.540298
  SBS_simple : 0.539900 (0, 1, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22)
  SBS_feature: 0.540342 (3, 5, 6, 7, 8, 10, 11, 12, 14, 15, 38, 44, 53, 55, 56, 64, 72, 86, 87)




"""
