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

train['k'] = (train['ap_hi'] - train['ap_lo']) / train['ap_lo']
train['h/w'] = train['height'] / train['weight']

# train['delta_h'] = np.round(train['height'].mean() - train['height'])
# train['delta_w'] = np.round(train['weight'].mean() - train['weight'])

train['bmi'] = np.round(train['weight'] / (train['height'] / 100) ** 2, 1)
train['obesity'] = train['bmi'].apply(lambda x: 1 if x >= 30 else 0)

print(train[:3])

print(train[(train['ch_3'] == 1) & (train['ap_hi'] > 160) & (train['gl_1'] == 1)]['cardio'].mean())

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
#
# model_t = xgb.train(params_est, d_train, len(model))
# plot_importance(model_t)
# plt.show()
#
# """
# ANSWER module
# """
# print('Answer module started')
# y_predict = model_t.predict(d_test)
# print(y_predict)
# pd.Series(y_predict).to_csv('answer.csv', index=False)
