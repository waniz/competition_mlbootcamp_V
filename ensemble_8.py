import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import xgboost as xgb
from FEATURES import AddFeatures

pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
plt.style.use('ggplot')
warnings.filterwarnings('ignore')
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

best_columns_first = [
    'gender', 'height', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y',
    'ch_1', 'ch_2', 'ch_3', 'gl_1', 'gl_2', 'gl_3',
    'bmi', 'sist_formula', 'map', 'F_score', 'ap_log'
]

clf_0_params = {
    'learning_rate': 0.007,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.73,
    'reg_lambda': 0.73,
    'gamma': 0.126,
    'min_child_weight': 90.8,
    'colsample_bytree': 0.89,
    'n_estimators': 821,
}
clf_0 = xgb.XGBClassifier(**clf_0_params)
print('Train clf_0')
clf_0.fit(X[best_columns_first], Y)

clf_1_params = {
    'learning_rate': 0.0074,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.72,
    'reg_lambda': 0.75,
    'gamma': 0.159,
    'min_child_weight': 85.5,
    'colsample_bytree': 0.84,
    'n_estimators': 825,
}
clf_1 = xgb.XGBClassifier(**clf_1_params)
print('Train clf_1')
clf_1.fit(X[best_columns_first], Y)

clf_2_params = {
    'learning_rate': 0.0021,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.6,
    'reg_lambda': 0.89,
    'gamma': 0.1,
    'min_child_weight': 49.8,
    'colsample_bytree': 0.8,
    'n_estimators': 2790,
}
clf_2 = xgb.XGBClassifier(**clf_2_params)
print('Train clf_2')
clf_2.fit(X[best_columns_first], Y)

clf_3_params = {
    'learning_rate': 0.0065,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.621,
    'reg_lambda': 0.726,
    'gamma': 0.053,
    'min_child_weight': 30.8,
    'colsample_bytree': 0.905,
    'n_estimators': 958,
}
clf_3 = xgb.XGBClassifier(**clf_3_params)
print('Train clf_3')
clf_3.fit(X[best_columns_first], Y)

clf_4_params = {
    'learning_rate': 0.0068,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.69,
    'reg_lambda': 0.88,
    'gamma': 0.15,
    'min_child_weight': 67,
    'colsample_bytree': 0.77,
    'n_estimators': 904,
}
clf_4 = xgb.XGBClassifier(**clf_4_params)
print('Train clf_4')
clf_4.fit(X[best_columns_first], Y)


best_columns_second = [
    'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
    'age_y', 'bmi', 'sist_formula', 'map', 'F_score'
]

clf_5_params = {
    'learning_rate': 0.0041,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.81,
    'reg_lambda': 0.94,
    'gamma': 0.115,
    'min_child_weight': 57,
    'colsample_bytree': 0.65,
    'n_estimators': 1559,
}
clf_5 = xgb.XGBClassifier(**clf_5_params)
print('Train clf_5')
clf_5.fit(X[best_columns_second], Y)

clf_6_params = {
    'learning_rate': 0.0046,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.77,
    'reg_lambda': 0.81,
    'gamma': 0.123,
    'min_child_weight': 54,
    'colsample_bytree': 0.72,
    'n_estimators': 1372,
}
clf_6 = xgb.XGBClassifier(**clf_6_params)
print('Train clf_6')
clf_6.fit(X[best_columns_second], Y)

clf_7_params = {
    'learning_rate': 0.0029,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.62,
    'reg_lambda': 0.75,
    'gamma': 0.129,
    'min_child_weight': 63,
    'colsample_bytree': 0.69,
    'n_estimators': 1990,
}
clf_7 = xgb.XGBClassifier(**clf_7_params)
print('Train clf_7')
clf_7.fit(X[best_columns_second], Y)

clf_8_params = {
    'learning_rate': 0.0036,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.71,
    'reg_lambda': 0.83,
    'gamma': 0.145,
    'min_child_weight': 11,
    'colsample_bytree': 0.69,
    'n_estimators': 1676,
}
clf_8 = xgb.XGBClassifier(**clf_8_params)
print('Train clf_8')
clf_8.fit(X[best_columns_second], Y)

clf_9_params = {
    'learning_rate': 0.0025,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.7,
    'reg_lambda': 0.85,
    'gamma': 0.196,
    'min_child_weight': 35,
    'colsample_bytree': 0.69,
    'n_estimators': 2351,
}
clf_9 = xgb.XGBClassifier(**clf_9_params)
print('Train clf_9')
clf_9.fit(X[best_columns_second], Y)

clf_10_params = {
    'learning_rate': 0.0069,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.82,
    'reg_lambda': 0.66,
    'gamma': 0.082,
    'min_child_weight': 35,
    'colsample_bytree': 0.67,
    'n_estimators': 850,
}
clf_10 = xgb.XGBClassifier(**clf_10_params)
print('Train clf_10')
clf_10.fit(X[best_columns_second], Y)

clf_11_params = {
    'learning_rate': 0.0094,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.76,
    'reg_lambda': 0.77,
    'gamma': 0.221,
    'min_child_weight': 76,
    'colsample_bytree': 0.71,
    'n_estimators': 517,
}
clf_11 = xgb.XGBClassifier(**clf_11_params)
print('Train clf_11')
clf_11.fit(X[best_columns_second], Y)

clf_12_params = {
    'learning_rate': 0.0086,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.84,
    'reg_lambda': 0.82,
    'gamma': 0.047,
    'min_child_weight': 29,
    'colsample_bytree': 0.82,
    'n_estimators': 716,
}
clf_12 = xgb.XGBClassifier(**clf_12_params)
print('Train clf_12')
clf_12.fit(X[best_columns_second], Y)

clf_13_params = {
    'learning_rate': 0.0084,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.64,
    'reg_lambda': 0.76,
    'gamma': 0.119,
    'min_child_weight': 77,
    'colsample_bytree': 0.87,
    'n_estimators': 733,
}
clf_13 = xgb.XGBClassifier(**clf_13_params)
print('Train clf_12')
clf_13.fit(X[best_columns_first], Y)


print('Answers generating...')
answers = pd.DataFrame()
answers['0'] = clf_0.predict_proba(test[best_columns_first])[:, 1]
answers['1'] = clf_1.predict_proba(test[best_columns_first])[:, 1]
answers['2'] = clf_2.predict_proba(test[best_columns_first])[:, 1]
answers['3'] = clf_3.predict_proba(test[best_columns_first])[:, 1]
answers['4'] = clf_4.predict_proba(test[best_columns_first])[:, 1]
answers['5'] = clf_5.predict_proba(test[best_columns_second])[:, 1]
answers['6'] = clf_6.predict_proba(test[best_columns_second])[:, 1]
answers['7'] = clf_7.predict_proba(test[best_columns_second])[:, 1]
answers['8'] = clf_8.predict_proba(test[best_columns_second])[:, 1]
answers['9'] = clf_9.predict_proba(test[best_columns_second])[:, 1]
answers['10'] = clf_10.predict_proba(test[best_columns_second])[:, 1]
answers['11'] = clf_11.predict_proba(test[best_columns_second])[:, 1]
answers['12'] = clf_12.predict_proba(test[best_columns_second])[:, 1]
answers['13'] = clf_13.predict_proba(test[best_columns_first])[:, 1]

answers['mean'] = answers.mean(axis=1)
print(answers[:10])

pd.Series(answers['mean']).to_csv('answer.csv', index=False)
