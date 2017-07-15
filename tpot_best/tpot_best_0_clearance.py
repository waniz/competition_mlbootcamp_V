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

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

print(train.shape, test.shape)

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train.as_matrix()

exported_pipeline = make_pipeline(
    StandardScaler(),
    ExtraTreesClassifier(bootstrap=True, max_features=0.75, min_samples_leaf=18, min_samples_split=13, n_estimators=100)
)

exported_pipeline.fit(X, Y)


"""
ANSWER module
"""

print('Answer module started')
y_predict = exported_pipeline.predict_proba(test.as_matrix())
print(y_predict)
pd.Series(y_predict[:, 1]).to_csv('answer.csv', index=False)


















