import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
from FEATURES import AddFeatures
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
N_CPU = 4

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

add_features = AddFeatures(train, test)
add_features.add_bmi_sist_dist_map()
add_features.add_f_score()

train = add_features.train
test = add_features.test

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train.as_matrix()

"""
RF data section:
    raw data result:
"""

print('RF calculations...')
params_rf = {
    'n_estimators': 1689,
    'criterion': 'gini',
    'max_features': 0.29,
    'min_samples_split': 0.01,
    'min_samples_leaf': 1,
}

# pipeline = RandomForestClassifier(**params_rf)
# scores = cross_val_score(pipeline, X, Y, cv=7, scoring='log_loss', n_jobs=N_CPU)
# print('RF: ', scores, np.mean(scores), np.std(scores))

# print('ET calculations...')
# params_et = {
#     'n_estimators': 100,
#     'criterion': 'entropy',
#     'max_features': 0.5,
#     'min_samples_split': 2,
#     'min_samples_leaf': 1,
# }
# pipeline = ExtraTreesClassifier(**params_et)
# scores = cross_val_score(pipeline, X, Y, cv=7, scoring='log_loss', n_jobs=N_CPU)
# print('ET: ', scores, np.mean(scores), np.std(scores))

"""
ANSWER module
"""
print('Answer module started')
model = RandomForestClassifier(**params_rf)
model.fit(X, Y)
y_predict = model.predict_proba(test.as_matrix())
pd.Series(y_predict[:, 1]).to_csv('answer.csv', index=False)


















