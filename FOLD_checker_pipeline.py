import numpy as np
import pandas as pd
import warnings
import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss
from FEATURES import AddFeatures
from mlxtend.classifier import EnsembleVoteClassifier

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
np.random.seed(42)


class FoldChecker:

    def __init__(self, pipeline_):
        train = pd.read_csv('data/train.csv', index_col=0)
        test = pd.read_csv('data/test.csv', index_col=0)

        add_features = AddFeatures(train, test)
        add_features.add_bmi_sist_dist_map()
        add_features.add_f_score()
        add_features.add_ap_features()
        add_features.del_features()

        train = add_features.train
        test = add_features.test

        self.Y = train['cardio'].values
        train.drop('cardio', axis=1, inplace=True)
        self.X = train
        self.test = test

        self.pipeline = pipeline_
        self.model = None
        self.kf = None

        self.results = pd.DataFrame()

    def fold_7(self):
        self.kf = StratifiedKFold(n_splits=7, shuffle=True)
        self.kf.get_n_splits(self.X[self.X.columns], self.Y)

        for train_index, test_index in self.kf.split(self.X[self.X.columns], self.Y):
            x_train, x_test = self.X.as_matrix(self.X.columns)[train_index], \
                              self.X.as_matrix(self.X.columns)[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            return x_train, y_train, x_test, y_test

    def fold_30_public(self):
        self.kf = StratifiedShuffleSplit(n_splits=1, test_size=0.15, train_size=None)
        self.kf.get_n_splits(self.X[self.X.columns], self.Y)

        for train_index, test_index in self.kf.split(self.X[self.X.columns], self.Y):
            x_train, x_test = self.X.as_matrix(self.X.columns)[train_index], \
                              self.X.as_matrix(self.X.columns)[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            return x_train, y_train, x_test, y_test

    def fold_30_private(self):
        self.kf = StratifiedShuffleSplit(n_splits=1, test_size=0.29, train_size=None)
        self.kf.get_n_splits(self.X[self.X.columns], self.Y)

        for train_index, test_index in self.kf.split(self.X[self.X.columns], self.Y):
            x_train, x_test = self.X.as_matrix(self.X.columns)[train_index], \
                              self.X.as_matrix(self.X.columns)[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

            return x_train, y_train, x_test, y_test

    def get_scores_pipeline(self):

        scores = []
        for i in tqdm.trange(7, desc='  [FOLD 7]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_7()
            self.model = self.pipeline.fit(x_train, y_train)
            predits = self.model.predict_proba(x_test)
            scores.append(log_loss(y_test, predits[:, 1]))
            print('\n    [FOLD 7_%s] mean: %s, std: %s' % (i, np.mean(scores), np.std(scores)))
        pass
        print('\n  [FOLD 7] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))
        pass

        scores = []
        for i in tqdm.trange(30, desc='  [FOLD PUBLIC]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_30_public()
            self.model = self.pipeline.fit(x_train, y_train)
            predits = self.model.predict_proba(x_test)
            scores.append(log_loss(y_test, predits[:, 1]))
            print('\n    [FOLD PUBLIC_%s] mean: %s, std: %s' % (i, np.mean(scores), np.std(scores)))
        pass
        print('\n  [FOLD PUBLIC] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))

        scores = []
        for i in tqdm.trange(30, desc='  [FOLD PRIVATE]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_30_private()
            self.model = self.pipeline.fit(x_train, y_train)
            predits = self.model.predict_proba(x_test)
            scores.append(log_loss(y_test, predits[:, 1]))
            print('\n    [FOLD PRIVATE %s] mean: %s, std: %s' % (i, np.mean(scores), np.std(scores)))
        pass
        print('\n  [FOLD PRIVATE] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))


clf_0_params = {
    'learning_rate': 0.007,
    'max_depth': 5,
    'nthread': -1,
    'silent': 1,
    'subsample': 0.73,
    'reg_lambda': 0.73,
    'gamma': 0.126,
    'min_child_weight': 90.8,
    'colsample_bytree': 0.89,
    'n_estimators': 821,
}
clf_0 = xgb.XGBClassifier(**clf_0_params)

clf_1_params = {
    'learning_rate': 0.0074,
    'max_depth': 5,
    'nthread': -1,
    'silent': 1,
    'subsample': 0.72,
    'reg_lambda': 0.75,
    'gamma': 0.159,
    'min_child_weight': 85.5,
    'colsample_bytree': 0.84,
    'n_estimators': 825,
}
clf_1 = xgb.XGBClassifier(**clf_1_params)

clf_2_params = {
    'learning_rate': 0.0021,
    'max_depth': 5,
    'nthread': -1,
    'silent': 1,
    'subsample': 0.6,
    'reg_lambda': 0.89,
    'gamma': 0.1,
    'min_child_weight': 49.8,
    'colsample_bytree': 0.8,
    'n_estimators': 2790,
}
clf_2 = xgb.XGBClassifier(**clf_2_params)

clf_3_params = {
    'learning_rate': 0.0065,
    'max_depth': 5,
    'nthread': -1,
    'silent': 1,
    'subsample': 0.621,
    'reg_lambda': 0.726,
    'gamma': 0.053,
    'min_child_weight': 30.8,
    'colsample_bytree': 0.905,
    'n_estimators': 958,
}
clf_3 = xgb.XGBClassifier(**clf_3_params)

pipeline = EnsembleVoteClassifier(clfs=[clf_0, clf_1, clf_2, clf_3], weights=[1, 1, 1, 1], voting='soft')

check = FoldChecker(pipeline_=pipeline)
print(check.get_scores_pipeline())
