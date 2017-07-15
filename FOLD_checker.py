import numpy as np
import pandas as pd
import warnings
import tqdm
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss
from FEATURES import AddFeatures

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
np.random.seed(42)


class FoldChecker:

    def __init__(self, train=None, test=None):
        if not train.empty:
            train = pd.read_csv('data/train.csv', index_col=0)
        if not test.empty:
            test = pd.read_csv('data/test.csv', index_col=0)

        add_features = AddFeatures(train, test)
        add_features.add_bmi_sist_dist_map()
        add_features.add_f_score()
        add_features.add_ap_features()
        add_features.del_features()

        train = add_features.train
        test = add_features.test

        best_columns_second = [
            'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'age_y', 'bmi', 'sist_formula', 'map', 'F_score'
        ]

        self.Y = train['cardio'].values
        train.drop('cardio', axis=1, inplace=True)
        self.X = train[best_columns_second]
        self.test = test[best_columns_second]

        self.pipeline = None
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

    def get_scores_xgb(self, state_iter, rounds=None, params_est=None):
        fold_7, fold_30_public, fold_30_private = [], [], []

        if not state_iter:
            state_iter = 42
        if not rounds:
            rounds = 559
        if not params_est:
            params_est = {
                'learning_rate': 0.010804048308415474,
                'max_depth': 5,
                'nthread': 4,
                'eval_metric': 'logloss',
                'silent': 1,
                'subsample': 0.5763362672172597,
                'reg_lambda': 0.7832325379384827,
                'gamma': 0.08645899817813536,
                'min_child_weight': 21.92331452549159,
                'colsample_bytree': 0.8454869615818112,
            }

        scores = []
        for _ in tqdm.trange(7, desc='  [FOLD 7]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_7()
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_test = xgb.DMatrix(x_test)
            self.model = xgb.train(params_est, d_train, rounds)
            predits = self.model.predict(d_test)
            scores.append(log_loss(y_test, predits))
        pass
        print('\n  [FOLD 7] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))
        fold_7.append(np.mean(scores))
        fold_7.append(np.std(scores))
        fold_7.append(np.min(scores))
        fold_7.append(np.max(scores))
        pass

        scores = []
        for _ in tqdm.trange(30, desc='  [FOLD PUBLIC]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_30_public()
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_test = xgb.DMatrix(x_test)

            self.model = xgb.train(params_est, d_train, rounds)
            predits = self.model.predict(d_test)
            for pos in range(len(predits)):
                if predits[pos] >= 1:
                    predits[pos] = 0.999999

            scores.append(log_loss(y_test, predits))
        pass
        print('\n  [FOLD PUBLIC] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))
        fold_30_public.append(np.mean(scores))
        fold_30_public.append(np.std(scores))
        fold_30_public.append(np.min(scores))
        fold_30_public.append(np.max(scores))

        scores = []
        for _ in tqdm.trange(30, desc='  [FOLD PRIVATE]', leave=False):
            x_train, y_train, x_test, y_test = self.fold_30_private()
            d_train = xgb.DMatrix(x_train, label=y_train)
            d_test = xgb.DMatrix(x_test)

            self.model = xgb.train(params_est, d_train, rounds)
            predits = self.model.predict(d_test)
            for pos in range(len(predits)):
                if predits[pos] >= 1:
                    predits[pos] = 0.999999

            scores.append(log_loss(y_test, predits))
        pass
        print('\n  [FOLD PRIVATE] mean: %s, std: %s' % (np.mean(scores), np.std(scores)))
        fold_30_private.append(np.mean(scores))
        fold_30_private.append(np.std(scores))
        fold_30_private.append(np.min(scores))
        fold_30_private.append(np.max(scores))

        self.results['7_fold'] = fold_7
        self.results['fold_public'] = fold_30_public
        self.results['fold_private'] = fold_30_private
        return self.results
