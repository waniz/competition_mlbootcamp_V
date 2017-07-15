import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from FEATURES import AddFeatures
import lightgbm as gbm

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
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

print(train[:5])

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train

params_est = {
    # 'learning_rate': 0.013,
    # 'max_depth': 5,
    # 'nthread': 4,
    # 'eval_metric': 'logloss',
    # 'silent': 1,
    # 'subsample': 0.463,
    # 'reg_lambda': 0.715,
    # 'gamma': 0.01,
    # 'min_child_weight': 30.4,
    # 'colsample_bytree': 0.859,
}

d_train = gbm.Dataset(train, label=Y)
d_test = gbm.Dataset(test)
model = gbm.cv(params_est, d_train, 100, nfold=7, stratified=True)

print(model)

rounds = len(model)
model_t = gbm.train(params_est, d_train, rounds)

"""
ANSWER module
"""
print('Answer module started')

y_predict = model_t.predict(test.as_matrix())
print(y_predict)
pd.Series(y_predict).to_csv('answer.csv', index=False)
