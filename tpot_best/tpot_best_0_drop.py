import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)
N_CPU = 4

train = pd.read_csv('../original_data/train.csv', delimiter=';')
test = pd.read_csv('../original_data/test.csv', delimiter=';')

print(train.shape, test.shape)

"""
 Data transformation block
"""
train['age_y'] = train['age'] / 365
train = train[train['height'] > 130]
train = train[train['height'] < 200]
train = train[train['height'] > 30]
train = train[train['ap_hi'] > 80]
train = train[train['ap_hi'] < 210]
train = train[train['ap_lo'] > 30]
train = train[train['ap_lo'] < 210]

test['age_y'] = test['age'] / 365
print(train.shape)

Y = train['cardio'].values
train.drop(['cardio', 'smoke', 'alco', 'active'], axis=1, inplace=True)
X = train.as_matrix()

exported_pipeline = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=18, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(X, Y)


"""
ANSWER module
"""
print('Answer module started')
test = test.replace('None', 0)
test.drop(['smoke', 'alco', 'active'], axis=1, inplace=True)
y_predict = exported_pipeline.predict_proba(test.as_matrix())
print(y_predict)
pd.Series(y_predict[:, 1]).to_csv('../answer.csv', index=False)


















