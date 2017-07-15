import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
np.random.seed(42)

train = pd.read_csv('data/train.csv', index_col=0)
test = pd.read_csv('data/test.csv', index_col=0)

train['bmi'] = np.round(train['weight'] / (train['height'] / 100) ** 2, 1)
train['sist_formula'] = 109 + 0.5 * train['age_y'] + 0.1 * train['weight']
train['dist_formula'] = 63 + 0.1 * train['age_y'] + 0.15 * train['weight']
train['map'] = train['ap_lo'] + 0.33 * (train['ap_hi'] - train['ap_lo'])

test['bmi'] = np.round(test['weight'] / (test['height'] / 100) ** 2, 1)
test['sist_formula'] = 109 + 0.5 * test['age_y'] + 0.1 * test['weight']
test['dist_formula'] = 63 + 0.1 * test['age_y'] + 0.15 * test['weight']
test['map'] = test['ap_lo'] + 0.33 * (test['ap_hi'] - test['ap_lo'])

"""
https://en.wikipedia.org/wiki/Framingham_Risk_Score
"""

train['F_score_0'] = 0
train['F_score_1'] = 0
train['F_score_2'] = 0
train['F_score_3'] = 0

train.at[(train['age_y'] <= 34) & (train['gender'] == 1), 'F_score_0'] = -9
train.at[(train['age_y'] > 34) & (train['gender'] == 1) & (train['age_y'] <= 39), 'F_score_0'] = -4
train.at[(train['age_y'] >= 40) & (train['gender'] == 1) & (train['age_y'] <= 44), 'F_score_0'] = 0
train.at[(train['age_y'] >= 45) & (train['gender'] == 1) & (train['age_y'] <= 49), 'F_score_0'] = 3
train.at[(train['age_y'] >= 50) & (train['gender'] == 1) & (train['age_y'] <= 54), 'F_score_0'] = 6
train.at[(train['age_y'] >= 55) & (train['gender'] == 1) & (train['age_y'] <= 59), 'F_score_0'] = 8
train.at[(train['age_y'] >= 60) & (train['gender'] == 1) & (train['age_y'] <= 64), 'F_score_0'] = 10
train.at[(train['age_y'] >= 65) & (train['gender'] == 1) & (train['age_y'] <= 69), 'F_score_0'] = 11
train.at[(train['age_y'] >= 70) & (train['gender'] == 1) & (train['age_y'] <= 74), 'F_score_0'] = 12
train.at[(train['age_y'] >= 75) & (train['gender'] == 1), 'F_score_0'] = 13
train.at[(train['age_y'] <= 34) & (train['gender'] == 0), 'F_score_0'] = -7
train.at[(train['age_y'] > 34) & (train['gender'] == 0) & (train['age_y'] <= 39), 'F_score_0'] = -3
train.at[(train['age_y'] >= 40) & (train['gender'] == 0) & (train['age_y'] <= 44), 'F_score_0'] = 0
train.at[(train['age_y'] >= 45) & (train['gender'] == 0) & (train['age_y'] <= 49), 'F_score_0'] = 3
train.at[(train['age_y'] >= 50) & (train['gender'] == 0) & (train['age_y'] <= 54), 'F_score_0'] = 6
train.at[(train['age_y'] >= 55) & (train['gender'] == 0) & (train['age_y'] <= 59), 'F_score_0'] = 8
train.at[(train['age_y'] >= 60) & (train['gender'] == 0) & (train['age_y'] <= 64), 'F_score_0'] = 10
train.at[(train['age_y'] >= 65) & (train['gender'] == 0) & (train['age_y'] <= 69), 'F_score_0'] = 12
train.at[(train['age_y'] >= 70) & (train['gender'] == 0) & (train['age_y'] <= 74), 'F_score_0'] = 14
train.at[(train['age_y'] >= 75) & (train['gender'] == 0), 'F_score_0'] = 16

train.at[(train['age_y'] <= 39) & (train['gender'] == 0) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] <= 39) & (train['gender'] == 0) & (train['ch_2'] == 1), 'F_score_1'] = 8
train.at[(train['age_y'] <= 39) & (train['gender'] == 0) & (train['ch_3'] == 1), 'F_score_1'] = 13
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 0) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 0) & (train['ch_2'] == 1), 'F_score_1'] = 6
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 0) & (train['ch_3'] == 1), 'F_score_1'] = 10
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 0) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 0) & (train['ch_2'] == 1), 'F_score_1'] = 4
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 0) & (train['ch_3'] == 1), 'F_score_1'] = 7
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 0) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 0) & (train['ch_2'] == 1), 'F_score_1'] = 2
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 0) & (train['ch_3'] == 1), 'F_score_1'] = 4
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 0) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 0) & (train['ch_2'] == 1), 'F_score_1'] = 1
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 0) & (train['ch_3'] == 1), 'F_score_1'] = 2
train.at[(train['age_y'] <= 39) & (train['gender'] == 1) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] <= 39) & (train['gender'] == 1) & (train['ch_2'] == 1), 'F_score_1'] = 7
train.at[(train['age_y'] <= 39) & (train['gender'] == 1) & (train['ch_3'] == 1), 'F_score_1'] = 11
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 1) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 1) & (train['ch_2'] == 1), 'F_score_1'] = 5
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 1) & (train['ch_3'] == 1), 'F_score_1'] = 8
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 1) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 1) & (train['ch_2'] == 1), 'F_score_1'] = 3
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 1) & (train['ch_3'] == 1), 'F_score_1'] = 5
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 1) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 1) & (train['ch_2'] == 1), 'F_score_1'] = 1
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 1) & (train['ch_3'] == 1), 'F_score_1'] = 3
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 1) & (train['ch_1'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 1) & (train['ch_2'] == 1), 'F_score_1'] = 0
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 1) & (train['ch_3'] == 1), 'F_score_1'] = 1

train.at[(train['age_y'] >= 0) & (train['age_y'] <= 39) & (train['gender'] == 0) & (train['smoke'] == 1), 'F_score_2'] = 9
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 0) & (train['smoke'] == 1), 'F_score_2'] = 7
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 0) & (train['smoke'] == 1), 'F_score_2'] = 4
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 0) & (train['smoke'] == 1), 'F_score_2'] = 2
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 0) & (train['smoke'] == 1), 'F_score_2'] = 1
train.at[(train['age_y'] >= 0) & (train['age_y'] <= 39) & (train['gender'] == 1) & (train['smoke'] == 1), 'F_score_2'] = 8
train.at[(train['age_y'] >= 40) & (train['age_y'] <= 49) & (train['gender'] == 1) & (train['smoke'] == 1), 'F_score_2'] = 5
train.at[(train['age_y'] >= 50) & (train['age_y'] <= 59) & (train['gender'] == 1) & (train['smoke'] == 1), 'F_score_2'] = 3
train.at[(train['age_y'] >= 60) & (train['age_y'] <= 69) & (train['gender'] == 1) & (train['smoke'] == 1), 'F_score_2'] = 1
train.at[(train['age_y'] >= 70) & (train['age_y'] <= 79) & (train['gender'] == 1) & (train['smoke'] == 1), 'F_score_2'] = 1

train.at[(train['ap_hi'] >= 120) & (train['ap_hi'] < 130) & (train['gender'] == 0), 'F_score_3'] = 1
train.at[(train['ap_hi'] >= 130) & (train['ap_hi'] < 140) & (train['gender'] == 0), 'F_score_3'] = 2
train.at[(train['ap_hi'] >= 140) & (train['ap_hi'] < 160) & (train['gender'] == 0), 'F_score_3'] = 3
train.at[(train['ap_hi'] >= 160) & (train['gender'] == 0), 'F_score_3'] = 4
train.at[(train['ap_hi'] >= 120) & (train['ap_hi'] < 130) & (train['gender'] == 1), 'F_score_3'] = 0
train.at[(train['ap_hi'] >= 130) & (train['ap_hi'] < 140) & (train['gender'] == 1), 'F_score_3'] = 1
train.at[(train['ap_hi'] >= 140) & (train['ap_hi'] < 160) & (train['gender'] == 1), 'F_score_3'] = 1
train.at[(train['ap_hi'] >= 160) & (train['gender'] == 1), 'F_score_3'] = 2

train['F_score'] = train['F_score_0'] + train['F_score_1'] + train['F_score_2'] + train['F_score_3']
train.drop(['F_score_0', 'F_score_1', 'F_score_2', 'F_score_3'], axis=1, inplace=True)

print(train[:3])

test['F_score_0'] = 0
test['F_score_1'] = 0
test['F_score_2'] = 0
test['F_score_3'] = 0

test.at[(test['age_y'] <= 34) & (test['gender'] == 1), 'F_score_0'] = -9
test.at[(test['age_y'] > 34) & (test['gender'] == 1) & (test['age_y'] <= 39), 'F_score_0'] = -4
test.at[(test['age_y'] >= 40) & (test['gender'] == 1) & (test['age_y'] <= 44), 'F_score_0'] = 0
test.at[(test['age_y'] >= 45) & (test['gender'] == 1) & (test['age_y'] <= 49), 'F_score_0'] = 3
test.at[(test['age_y'] >= 50) & (test['gender'] == 1) & (test['age_y'] <= 54), 'F_score_0'] = 6
test.at[(test['age_y'] >= 55) & (test['gender'] == 1) & (test['age_y'] <= 59), 'F_score_0'] = 8
test.at[(test['age_y'] >= 60) & (test['gender'] == 1) & (test['age_y'] <= 64), 'F_score_0'] = 10
test.at[(test['age_y'] >= 65) & (test['gender'] == 1) & (test['age_y'] <= 69), 'F_score_0'] = 11
test.at[(test['age_y'] >= 70) & (test['gender'] == 1) & (test['age_y'] <= 74), 'F_score_0'] = 12
test.at[(test['age_y'] >= 75) & (test['gender'] == 1), 'F_score_0'] = 13
test.at[(test['age_y'] <= 34) & (test['gender'] == 0), 'F_score_0'] = -7
test.at[(test['age_y'] > 34) & (test['gender'] == 0) & (test['age_y'] <= 39), 'F_score_0'] = -3
test.at[(test['age_y'] >= 40) & (test['gender'] == 0) & (test['age_y'] <= 44), 'F_score_0'] = 0
test.at[(test['age_y'] >= 45) & (test['gender'] == 0) & (test['age_y'] <= 49), 'F_score_0'] = 3
test.at[(test['age_y'] >= 50) & (test['gender'] == 0) & (test['age_y'] <= 54), 'F_score_0'] = 6
test.at[(test['age_y'] >= 55) & (test['gender'] == 0) & (test['age_y'] <= 59), 'F_score_0'] = 8
test.at[(test['age_y'] >= 60) & (test['gender'] == 0) & (test['age_y'] <= 64), 'F_score_0'] = 10
test.at[(test['age_y'] >= 65) & (test['gender'] == 0) & (test['age_y'] <= 69), 'F_score_0'] = 12
test.at[(test['age_y'] >= 70) & (test['gender'] == 0) & (test['age_y'] <= 74), 'F_score_0'] = 14
test.at[(test['age_y'] >= 75) & (test['gender'] == 0), 'F_score_0'] = 16

test.at[(test['age_y'] <= 39) & (test['gender'] == 0) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] <= 39) & (test['gender'] == 0) & (test['ch_2'] == 1), 'F_score_1'] = 8
test.at[(test['age_y'] <= 39) & (test['gender'] == 0) & (test['ch_3'] == 1), 'F_score_1'] = 13
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 0) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 0) & (test['ch_2'] == 1), 'F_score_1'] = 6
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 0) & (test['ch_3'] == 1), 'F_score_1'] = 10
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 0) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 0) & (test['ch_2'] == 1), 'F_score_1'] = 4
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 0) & (test['ch_3'] == 1), 'F_score_1'] = 7
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 0) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 0) & (test['ch_2'] == 1), 'F_score_1'] = 2
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 0) & (test['ch_3'] == 1), 'F_score_1'] = 4
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 0) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 0) & (test['ch_2'] == 1), 'F_score_1'] = 1
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 0) & (test['ch_3'] == 1), 'F_score_1'] = 2
test.at[(test['age_y'] <= 39) & (test['gender'] == 1) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] <= 39) & (test['gender'] == 1) & (test['ch_2'] == 1), 'F_score_1'] = 7
test.at[(test['age_y'] <= 39) & (test['gender'] == 1) & (test['ch_3'] == 1), 'F_score_1'] = 11
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 1) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 1) & (test['ch_2'] == 1), 'F_score_1'] = 5
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 1) & (test['ch_3'] == 1), 'F_score_1'] = 8
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 1) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 1) & (test['ch_2'] == 1), 'F_score_1'] = 3
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 1) & (test['ch_3'] == 1), 'F_score_1'] = 5
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 1) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 1) & (test['ch_2'] == 1), 'F_score_1'] = 1
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 1) & (test['ch_3'] == 1), 'F_score_1'] = 3
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 1) & (test['ch_1'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 1) & (test['ch_2'] == 1), 'F_score_1'] = 0
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 1) & (test['ch_3'] == 1), 'F_score_1'] = 1

test.at[(test['age_y'] >= 0) & (test['age_y'] <= 39) & (test['gender'] == 0) & (test['smoke'] == 1), 'F_score_2'] = 9
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 0) & (test['smoke'] == 1), 'F_score_2'] = 7
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 0) & (test['smoke'] == 1), 'F_score_2'] = 4
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 0) & (test['smoke'] == 1), 'F_score_2'] = 2
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 0) & (test['smoke'] == 1), 'F_score_2'] = 1
test.at[(test['age_y'] >= 0) & (test['age_y'] <= 39) & (test['gender'] == 1) & (test['smoke'] == 1), 'F_score_2'] = 8
test.at[(test['age_y'] >= 40) & (test['age_y'] <= 49) & (test['gender'] == 1) & (test['smoke'] == 1), 'F_score_2'] = 5
test.at[(test['age_y'] >= 50) & (test['age_y'] <= 59) & (test['gender'] == 1) & (test['smoke'] == 1), 'F_score_2'] = 3
test.at[(test['age_y'] >= 60) & (test['age_y'] <= 69) & (test['gender'] == 1) & (test['smoke'] == 1), 'F_score_2'] = 1
test.at[(test['age_y'] >= 70) & (test['age_y'] <= 79) & (test['gender'] == 1) & (test['smoke'] == 1), 'F_score_2'] = 1

test.at[(test['ap_hi'] >= 120) & (test['ap_hi'] < 130) & (test['gender'] == 0), 'F_score_3'] = 1
test.at[(test['ap_hi'] >= 130) & (test['ap_hi'] < 140) & (test['gender'] == 0), 'F_score_3'] = 2
test.at[(test['ap_hi'] >= 140) & (test['ap_hi'] < 160) & (test['gender'] == 0), 'F_score_3'] = 3
test.at[(test['ap_hi'] >= 160) & (test['gender'] == 0), 'F_score_3'] = 4
test.at[(test['ap_hi'] >= 120) & (test['ap_hi'] < 130) & (test['gender'] == 1), 'F_score_3'] = 0
test.at[(test['ap_hi'] >= 130) & (test['ap_hi'] < 140) & (test['gender'] == 1), 'F_score_3'] = 1
test.at[(test['ap_hi'] >= 140) & (test['ap_hi'] < 160) & (test['gender'] == 1), 'F_score_3'] = 1
test.at[(test['ap_hi'] >= 160) & (test['gender'] == 1), 'F_score_3'] = 2

test['F_score'] = test['F_score_0'] + test['F_score_1'] + test['F_score_2'] + test['F_score_3']
test.drop(['F_score_0', 'F_score_1', 'F_score_2', 'F_score_3'], axis=1, inplace=True)


Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train

params_est = {
    'learning_rate': 0.02,
    'max_depth': 6,
    'nthread': 4,
    'subsample': 0.6,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'silent': 1,
}

d_train = xgb.DMatrix(train, label=Y)
d_test = xgb.DMatrix(test)
model = xgb.cv(params_est, d_train, 600, nfold=7, stratified=True, early_stopping_rounds=50)
print(min(model['test-logloss-mean']), np.std(model['test-logloss-mean']), len(model))

model_t = xgb.train(params_est, d_train, len(model))
plot_importance(model_t)
plt.show()

"""
ANSWER module
"""
print('Answer module started')
y_predict = model_t.predict(d_test)
print(y_predict)
pd.Series(y_predict).to_csv('answer.csv', index=False)
