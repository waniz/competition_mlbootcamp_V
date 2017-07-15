import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
import warnings
from FEATURES import AddFeatures
import seaborn as sns
from sklearn.model_selection import train_test_split


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

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train

# best_columns = [
#     'gender', 'height', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y',
#     'ch_1', 'ch_2', 'ch_3', 'gl_1', 'gl_2', 'gl_3',
#     'bmi', 'sist_formula', 'map', 'F_score', 'ap_log'
# ]

best_columns = [
    'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'age_y', 'bmi', 'sist_formula', 'map', 'F_score'
]


params_est = {
    'learning_rate': 0.0015,
    'max_depth': 6,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.54,
    'reg_lambda': 0.6,
    'gamma': 0.041,
    'min_child_weight': 100,
    'colsample_bytree': 0.78,
    'n_estimators': 3140,
}

d_train = xgb.DMatrix(train[best_columns], label=Y)
d_test = xgb.DMatrix(test[best_columns])

rounds = 1559

model_t = xgb.train(params_est, d_train, rounds)
plot_importance(model_t)
plt.show()

"""
ANSWER module
"""
print('Answer module started')
y_predict = model_t.predict(d_test)
print(y_predict)
pd.Series(y_predict).to_csv('answer.csv', index=False)
