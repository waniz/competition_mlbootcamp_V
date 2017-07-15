import warnings
import numpy as np
import pandas as pd
from tpot import TPOTClassifier
from FEATURES import AddFeatures

warnings.filterwarnings('ignore')
np.random.seed(42)
pd.set_option('display.max_columns', 16)
pd.set_option('display.width', 1000)

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
X = train.as_matrix()

pipeline_optimizer = TPOTClassifier(
    generations=16, population_size=36,
    scoring='log_loss', cv=5, n_jobs=36, random_state=42,
    verbosity=2, max_eval_time_mins=2,
)

pipeline_optimizer.fit(X, Y)
pipeline_optimizer.export('export_server.py')
