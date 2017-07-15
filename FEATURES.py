import pandas as pd
import numpy as np


class AddFeatures:

    def __init__(self, train, test):
        self.train = train.copy()
        self.test = test.copy()

    def add_bmi_sist_dist_map(self):
        self.train['bmi'] = np.round(self.train['weight'] / (self.train['height'] / 100) ** 2, 1)
        self.train['sist_formula'] = 109 + 0.5 * self.train['age_y'] + 0.1 * self.train['weight']
        self.train['dist_formula'] = 63 + 0.1 * self.train['age_y'] + 0.15 * self.train['weight']
        self.train['map'] = self.train['ap_lo'] + 0.33 * (self.train['ap_hi'] -self.train['ap_lo'])
        self.test['bmi'] = np.round(self.test['weight'] / (self.test['height'] / 100) ** 2, 1)
        self.test['sist_formula'] = 109 + 0.5 * self.test['age_y'] + 0.1 * self.test['weight']
        self.test['dist_formula'] = 63 + 0.1 * self.test['age_y'] + 0.15 * self.test['weight']
        self.test['map'] = self.test['ap_lo'] + 0.33 * (self.test['ap_hi'] - self.test['ap_lo'])

    def add_f_score(self):
        self.train['F_score_0'] = 0
        self.train['F_score_1'] = 0
        self.train['F_score_2'] = 0
        self.train['F_score_3'] = 0

        self.train.at[(self.train['age_y'] <= 34) & (self.train['gender'] == 1), 'F_score_0'] = -9
        self.train.at[
            (self.train['age_y'] > 34) & (self.train['gender'] == 1) & (self.train['age_y'] <= 39), 'F_score_0'] = -4
        self.train.at[
            (self.train['age_y'] >= 40) & (self.train['gender'] == 1) & (self.train['age_y'] <= 44), 'F_score_0'] = 0
        self.train.at[
            (self.train['age_y'] >= 45) & (self.train['gender'] == 1) & (self.train['age_y'] <= 49), 'F_score_0'] = 3
        self.train.at[
            (self.train['age_y'] >= 50) & (self.train['gender'] == 1) & (self.train['age_y'] <= 54), 'F_score_0'] = 6
        self.train.at[
            (self.train['age_y'] >= 55) & (self.train['gender'] == 1) & (self.train['age_y'] <= 59), 'F_score_0'] = 8
        self.train.at[
            (self.train['age_y'] >= 60) & (self.train['gender'] == 1) & (self.train['age_y'] <= 64), 'F_score_0'] = 10
        self.train.at[
            (self.train['age_y'] >= 65) & (self.train['gender'] == 1) & (self.train['age_y'] <= 69), 'F_score_0'] = 11
        self.train.at[
            (self.train['age_y'] >= 70) & (self.train['gender'] == 1) & (self.train['age_y'] <= 74), 'F_score_0'] = 12
        self.train.at[(self.train['age_y'] >= 75) & (self.train['gender'] == 1), 'F_score_0'] = 13
        self.train.at[(self.train['age_y'] <= 34) & (self.train['gender'] == 0), 'F_score_0'] = -7
        self.train.at[
            (self.train['age_y'] > 34) & (self.train['gender'] == 0) & (self.train['age_y'] <= 39), 'F_score_0'] = -3
        self.train.at[
            (self.train['age_y'] >= 40) & (self.train['gender'] == 0) & (self.train['age_y'] <= 44), 'F_score_0'] = 0
        self.train.at[
            (self.train['age_y'] >= 45) & (self.train['gender'] == 0) & (self.train['age_y'] <= 49), 'F_score_0'] = 3
        self.train.at[
            (self.train['age_y'] >= 50) & (self.train['gender'] == 0) & (self.train['age_y'] <= 54), 'F_score_0'] = 6
        self.train.at[
            (self.train['age_y'] >= 55) & (self.train['gender'] == 0) & (self.train['age_y'] <= 59), 'F_score_0'] = 8
        self.train.at[
            (self.train['age_y'] >= 60) & (self.train['gender'] == 0) & (self.train['age_y'] <= 64), 'F_score_0'] = 10
        self.train.at[
            (self.train['age_y'] >= 65) & (self.train['gender'] == 0) & (self.train['age_y'] <= 69), 'F_score_0'] = 12
        self.train.at[
            (self.train['age_y'] >= 70) & (self.train['gender'] == 0) & (self.train['age_y'] <= 74), 'F_score_0'] = 14
        self.train.at[(self.train['age_y'] >= 75) & (self.train['gender'] == 0), 'F_score_0'] = 16

        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 0) & (self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 0) & (self.train['ch_2'] == 1), 'F_score_1'] = 8
        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 0) & (self.train['ch_3'] == 1), 'F_score_1'] = 13
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 0) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 0) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 6
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 0) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 10
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 0) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 0) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 4
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 0) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 7
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 0) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 0) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 2
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 0) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 4
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 0) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 0) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 1
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 0) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 2
        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 1) & (self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 1) & (self.train['ch_2'] == 1), 'F_score_1'] = 7
        self.train.at[
            (self.train['age_y'] <= 39) & (self.train['gender'] == 1) & (self.train['ch_3'] == 1), 'F_score_1'] = 11
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 1) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 1) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 5
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 1) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 8
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 1) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 1) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 3
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 1) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 5
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 1) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 1) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 1
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 1) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 3
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 1) & (
            self.train['ch_1'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 1) & (
            self.train['ch_2'] == 1), 'F_score_1'] = 0
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 1) & (
            self.train['ch_3'] == 1), 'F_score_1'] = 1

        self.train.at[(self.train['age_y'] >= 0) & (self.train['age_y'] <= 39) & (self.train['gender'] == 0) & (
            self.train['smoke'] == 1), 'F_score_2'] = 9
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 0) & (
            self.train['smoke'] == 1), 'F_score_2'] = 7
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 0) & (
            self.train['smoke'] == 1), 'F_score_2'] = 4
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 0) & (
            self.train['smoke'] == 1), 'F_score_2'] = 2
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 0) & (
            self.train['smoke'] == 1), 'F_score_2'] = 1
        self.train.at[(self.train['age_y'] >= 0) & (self.train['age_y'] <= 39) & (self.train['gender'] == 1) & (
            self.train['smoke'] == 1), 'F_score_2'] = 8
        self.train.at[(self.train['age_y'] >= 40) & (self.train['age_y'] <= 49) & (self.train['gender'] == 1) & (
            self.train['smoke'] == 1), 'F_score_2'] = 5
        self.train.at[(self.train['age_y'] >= 50) & (self.train['age_y'] <= 59) & (self.train['gender'] == 1) & (
            self.train['smoke'] == 1), 'F_score_2'] = 3
        self.train.at[(self.train['age_y'] >= 60) & (self.train['age_y'] <= 69) & (self.train['gender'] == 1) & (
            self.train['smoke'] == 1), 'F_score_2'] = 1
        self.train.at[(self.train['age_y'] >= 70) & (self.train['age_y'] <= 79) & (self.train['gender'] == 1) & (
            self.train['smoke'] == 1), 'F_score_2'] = 1

        self.train.at[
            (self.train['ap_hi'] >= 120) & (self.train['ap_hi'] < 130) & (self.train['gender'] == 0), 'F_score_3'] = 1
        self.train.at[
            (self.train['ap_hi'] >= 130) & (self.train['ap_hi'] < 140) & (self.train['gender'] == 0), 'F_score_3'] = 2
        self.train.at[
            (self.train['ap_hi'] >= 140) & (self.train['ap_hi'] < 160) & (self.train['gender'] == 0), 'F_score_3'] = 3
        self.train.at[(self.train['ap_hi'] >= 160) & (self.train['gender'] == 0), 'F_score_3'] = 4
        self.train.at[
            (self.train['ap_hi'] >= 120) & (self.train['ap_hi'] < 130) & (self.train['gender'] == 1), 'F_score_3'] = 0
        self.train.at[
            (self.train['ap_hi'] >= 130) & (self.train['ap_hi'] < 140) & (self.train['gender'] == 1), 'F_score_3'] = 1
        self.train.at[
            (self.train['ap_hi'] >= 140) & (self.train['ap_hi'] < 160) & (self.train['gender'] == 1), 'F_score_3'] = 1
        self.train.at[(self.train['ap_hi'] >= 160) & (self.train['gender'] == 1), 'F_score_3'] = 2

        self.train['F_score'] = self.train['F_score_0'] + self.train['F_score_1'] + self.train['F_score_2'] + \
                                self.train['F_score_3']
        self.train.drop(['F_score_0', 'F_score_1', 'F_score_2', 'F_score_3'], axis=1, inplace=True)

        self.test['F_score_0'] = 0
        self.test['F_score_1'] = 0
        self.test['F_score_2'] = 0
        self.test['F_score_3'] = 0

        self.test.at[(self.test['age_y'] <= 34) & (self.test['gender'] == 1), 'F_score_0'] = -9
        self.test.at[
            (self.test['age_y'] > 34) & (self.test['gender'] == 1) & (self.test['age_y'] <= 39), 'F_score_0'] = -4
        self.test.at[
            (self.test['age_y'] >= 40) & (self.test['gender'] == 1) & (self.test['age_y'] <= 44), 'F_score_0'] = 0
        self.test.at[
            (self.test['age_y'] >= 45) & (self.test['gender'] == 1) & (self.test['age_y'] <= 49), 'F_score_0'] = 3
        self.test.at[
            (self.test['age_y'] >= 50) & (self.test['gender'] == 1) & (self.test['age_y'] <= 54), 'F_score_0'] = 6
        self.test.at[
            (self.test['age_y'] >= 55) & (self.test['gender'] == 1) & (self.test['age_y'] <= 59), 'F_score_0'] = 8
        self.test.at[
            (self.test['age_y'] >= 60) & (self.test['gender'] == 1) & (self.test['age_y'] <= 64), 'F_score_0'] = 10
        self.test.at[
            (self.test['age_y'] >= 65) & (self.test['gender'] == 1) & (self.test['age_y'] <= 69), 'F_score_0'] = 11
        self.test.at[
            (self.test['age_y'] >= 70) & (self.test['gender'] == 1) & (self.test['age_y'] <= 74), 'F_score_0'] = 12
        self.test.at[(self.test['age_y'] >= 75) & (self.test['gender'] == 1), 'F_score_0'] = 13
        self.test.at[(self.test['age_y'] <= 34) & (self.test['gender'] == 0), 'F_score_0'] = -7
        self.test.at[
            (self.test['age_y'] > 34) & (self.test['gender'] == 0) & (self.test['age_y'] <= 39), 'F_score_0'] = -3
        self.test.at[
            (self.test['age_y'] >= 40) & (self.test['gender'] == 0) & (self.test['age_y'] <= 44), 'F_score_0'] = 0
        self.test.at[
            (self.test['age_y'] >= 45) & (self.test['gender'] == 0) & (self.test['age_y'] <= 49), 'F_score_0'] = 3
        self.test.at[
            (self.test['age_y'] >= 50) & (self.test['gender'] == 0) & (self.test['age_y'] <= 54), 'F_score_0'] = 6
        self.test.at[
            (self.test['age_y'] >= 55) & (self.test['gender'] == 0) & (self.test['age_y'] <= 59), 'F_score_0'] = 8
        self.test.at[
            (self.test['age_y'] >= 60) & (self.test['gender'] == 0) & (self.test['age_y'] <= 64), 'F_score_0'] = 10
        self.test.at[
            (self.test['age_y'] >= 65) & (self.test['gender'] == 0) & (self.test['age_y'] <= 69), 'F_score_0'] = 12
        self.test.at[
            (self.test['age_y'] >= 70) & (self.test['gender'] == 0) & (self.test['age_y'] <= 74), 'F_score_0'] = 14
        self.test.at[(self.test['age_y'] >= 75) & (self.test['gender'] == 0), 'F_score_0'] = 16

        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 0) & (self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 0) & (self.test['ch_2'] == 1), 'F_score_1'] = 8
        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 0) & (self.test['ch_3'] == 1), 'F_score_1'] = 13
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 0) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 0) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 6
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 0) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 10
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 0) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 0) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 4
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 0) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 7
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 0) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 0) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 2
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 0) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 4
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 0) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 0) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 1
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 0) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 2
        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 1) & (self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 1) & (self.test['ch_2'] == 1), 'F_score_1'] = 7
        self.test.at[
            (self.test['age_y'] <= 39) & (self.test['gender'] == 1) & (self.test['ch_3'] == 1), 'F_score_1'] = 11
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 1) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 1) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 5
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 1) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 8
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 1) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 1) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 3
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 1) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 5
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 1) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 1) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 1
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 1) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 3
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 1) & (
            self.test['ch_1'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 1) & (
            self.test['ch_2'] == 1), 'F_score_1'] = 0
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 1) & (
            self.test['ch_3'] == 1), 'F_score_1'] = 1

        self.test.at[(self.test['age_y'] >= 0) & (self.test['age_y'] <= 39) & (self.test['gender'] == 0) & (
            self.test['smoke'] == 1), 'F_score_2'] = 9
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 0) & (
            self.test['smoke'] == 1), 'F_score_2'] = 7
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 0) & (
            self.test['smoke'] == 1), 'F_score_2'] = 4
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 0) & (
            self.test['smoke'] == 1), 'F_score_2'] = 2
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 0) & (
            self.test['smoke'] == 1), 'F_score_2'] = 1
        self.test.at[(self.test['age_y'] >= 0) & (self.test['age_y'] <= 39) & (self.test['gender'] == 1) & (
            self.test['smoke'] == 1), 'F_score_2'] = 8
        self.test.at[(self.test['age_y'] >= 40) & (self.test['age_y'] <= 49) & (self.test['gender'] == 1) & (
            self.test['smoke'] == 1), 'F_score_2'] = 5
        self.test.at[(self.test['age_y'] >= 50) & (self.test['age_y'] <= 59) & (self.test['gender'] == 1) & (
            self.test['smoke'] == 1), 'F_score_2'] = 3
        self.test.at[(self.test['age_y'] >= 60) & (self.test['age_y'] <= 69) & (self.test['gender'] == 1) & (
            self.test['smoke'] == 1), 'F_score_2'] = 1
        self.test.at[(self.test['age_y'] >= 70) & (self.test['age_y'] <= 79) & (self.test['gender'] == 1) & (
            self.test['smoke'] == 1), 'F_score_2'] = 1

        self.test.at[
            (self.test['ap_hi'] >= 120) & (self.test['ap_hi'] < 130) & (self.test['gender'] == 0), 'F_score_3'] = 1
        self.test.at[
            (self.test['ap_hi'] >= 130) & (self.test['ap_hi'] < 140) & (self.test['gender'] == 0), 'F_score_3'] = 2
        self.test.at[
            (self.test['ap_hi'] >= 140) & (self.test['ap_hi'] < 160) & (self.test['gender'] == 0), 'F_score_3'] = 3
        self.test.at[(self.test['ap_hi'] >= 160) & (self.test['gender'] == 0), 'F_score_3'] = 4
        self.test.at[
            (self.test['ap_hi'] >= 120) & (self.test['ap_hi'] < 130) & (self.test['gender'] == 1), 'F_score_3'] = 0
        self.test.at[
            (self.test['ap_hi'] >= 130) & (self.test['ap_hi'] < 140) & (self.test['gender'] == 1), 'F_score_3'] = 1
        self.test.at[
            (self.test['ap_hi'] >= 140) & (self.test['ap_hi'] < 160) & (self.test['gender'] == 1), 'F_score_3'] = 1
        self.test.at[(self.test['ap_hi'] >= 160) & (self.test['gender'] == 1), 'F_score_3'] = 2

        self.test['F_score'] = self.test['F_score_0'] + self.test['F_score_1'] + self.test['F_score_2'] + self.test[
            'F_score_3']
        self.test.drop(['F_score_0', 'F_score_1', 'F_score_2', 'F_score_3'], axis=1, inplace=True)

    def add_ap_features(self):
        self.train['ap_log'] = np.log(self.train['ap_hi'])
        self.train['ap_/'] = self.train['ap_hi'] / (self.train['ap_lo'] + 1)
        self.train['ap_diff'] = abs(self.train['ap_diff'])
        self.train['k'] = (self.train['ap_hi'] - self.train['ap_lo']) / self.train['ap_lo']
        self.train['h/w'] = self.train['height'] / self.train['weight']
        self.train['obesity'] = self.train['bmi'].apply(lambda x: 1 if x >= 30 else 0)
        self.train['age_m'] = np.round(self.train['age'] / 30.4375)

        self.test['ap_log'] = np.log(self.test['ap_hi'])
        self.test['ap_/'] = self.test['ap_hi'] / (self.test['ap_lo'] + 1)
        self.test['ap_diff'] = abs(self.test['ap_diff'])
        self.test['k'] = (self.test['ap_hi'] - self.test['ap_lo']) / self.test['ap_lo']
        self.test['h/w'] = self.test['height'] / self.test['weight']
        self.test['obesity'] = self.test['bmi'].apply(lambda x: 1 if x >= 30 else 0)
        self.test['age_m'] = np.round(self.test['age'] / 30.4375)

    def del_features(self):
        del self.train['age']
        del self.test['age']



