import pandas as pd
import numpy as np
import xgboost as xgb
from matplotlib import pyplot as plt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from FEATURES import AddFeatures
import warnings

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
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

print(train.columns)


Y = train['cardio']
train.drop('cardio', axis=1, inplace=True)
X = train

params_est = {
    'learning_rate': 0.013,
    'max_depth': 5,
    'nthread': 4,
    'silent': 1,
    'subsample': 0.463,
    'reg_lambda': 0.715,
    'gamma': 0.01,
    'min_child_weight': 30.4,
}

estimator = xgb.XGBClassifier(**params_est)

sfs1 = SFS(estimator, k_features=(1, 26), forward=True, floating=False, verbose=2, scoring='log_loss', cv=4, n_jobs=4)
sfs1 = sfs1.fit(train.as_matrix(), Y)

results = pd.DataFrame.from_dict(sfs1.get_metric_dict()).T
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

print(sfs1.subsets_)
print(sfs1.k_feature_idx_)
print(sfs1.k_score_)


"""
results with fast code:
 (1, 3, 7, 8, 10, 11, 12, 14, 15, 17, 19, 20, 23, 24) = -0.616370076628

 ['height', 'ap_hi', 'active', 'age_y', 'ch_1', 'ch_2', 'ch_3', 'gl_2', 'gl_3', 'sist_formula', 'map', 'F_score',
  'ap_h_delta', 'ap_l_delta']

results with normal code:
 (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23) -0.556838879887

 ['gender', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y', 'ap_diff', 'ch_1', 'ch_2', 'ch_3', 'gl_1',
  'gl_2', 'gl_3', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log', 'ap_/']

"""