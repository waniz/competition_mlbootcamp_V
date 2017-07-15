import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)


train = pd.read_csv('original_data/train.csv', delimiter=';', index_col=0)
test = pd.read_csv('original_data/test.csv', delimiter=';', index_col=0)


def describe_more(df_):
    var = []
    l = []
    t = []
    for x in df_:
        var.append(x)
        l.append(len(pd.value_counts(df_[x])))
        t.append(df_[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels

"""
 set transformation
"""
train['age_y'] = np.floor(train['age'] / 365.25)
train['gender'] -= 1

train.at[train['height'] == 250, 'height'] = 150
train.loc[train['height'] < 100, 'height'] += 100
print('After HEIGHT clearance: ', train.shape)

train.at[train['weight'] == 13, 'weight'] = 73
train.at[train['weight'] == 10, 'weight'] = 70
train.at[train['weight'] == 30, 'weight'] = 80
train.at[train['weight'] == 16.3, 'weight'] = 130
train.at[train['weight'] == 20, 'weight'] = 120
train.loc[(train['weight'] > 30) & (train['weight'] < 40), 'weight'] += 50
train.at[train['weight'] == 23, 'weight'] = 73
train.at[train['weight'] == 22, 'weight'] = 72
train.at[train['weight'] == 11, 'weight'] = 77
train.at[train['weight'] == 28, 'weight'] = 78
train.at[train['weight'] == 21, 'weight'] = 71
train.at[train['weight'] == 29, 'weight'] = 79
print('After WEIGHT clearance: ', train.shape)

train.at[train['ap_hi'] == -100, 'ap_hi'] = 100
train.at[train['ap_hi'] == -115, 'ap_hi'] = 115
train.at[train['ap_hi'] == -140, 'ap_hi'] = 140
train.at[train['ap_hi'] == -120, 'ap_hi'] = 120
train.at[train['ap_hi'] == -150, 'ap_hi'] = 150
df = train[train['ap_hi'] <= 20]
train = train[train['ap_hi'] > 20]
df['ap_hi'] *= 10
train = train.append(df)
train.at[train.index == 75399, 'ap_hi'] = 240
train.at[train.index == 75399, 'ap_lo'] = 200
train.at[train.index == 12494, 'ap_hi'] = 120
train.at[train.index == 12494, 'ap_lo'] = 88
train.at[train.index == 60477, 'ap_hi'] = 110
train.at[train.index == 60477, 'ap_lo'] = 88

train.at[train.index == 2654, 'ap_hi'] = 90
train.at[train.index == 2654, 'ap_lo'] = 60
train.at[train.index == 2845, 'ap_hi'] = 90
train.at[train.index == 2845, 'ap_lo'] = 60
train.at[train.index == 6822, 'ap_hi'] = 90
train.at[train.index == 6822, 'ap_lo'] = 60
train.at[train.index == 11089, 'ap_hi'] = 115
train.at[train.index == 11089, 'ap_lo'] = 90
train.at[train.index == 12710, 'ap_hi'] = 142
train.at[train.index == 12710, 'ap_lo'] = 80
train.at[train.index == 13616, 'ap_hi'] = 170
train.at[train.index == 13616, 'ap_lo'] = 110
train.at[train.index == 19827, 'ap_hi'] = 150
train.at[train.index == 19827, 'ap_lo'] = 80
train.at[train.index == 25314, 'ap_hi'] = 107
train.at[train.index == 25314, 'ap_lo'] = 70
train.at[train.index == 34120, 'ap_hi'] = 90
train.at[train.index == 34120, 'ap_lo'] = 60
train.at[train.index == 36339, 'ap_hi'] = 140
train.at[train.index == 36339, 'ap_lo'] = 80
train.at[train.index == 36414, 'ap_hi'] = 140
train.at[train.index == 36414, 'ap_lo'] = 80
train.at[train.index == 36793, 'ap_hi'] = 140
train.at[train.index == 36793, 'ap_lo'] = 80
train.at[train.index == 40239, 'ap_hi'] = 160
train.at[train.index == 40239, 'ap_lo'] = 80
train.at[train.index == 42410, 'ap_hi'] = 90
train.at[train.index == 42410, 'ap_lo'] = 70
train.at[train.index == 45400, 'ap_hi'] = 90
train.at[train.index == 45400, 'ap_lo'] = 70
train.at[train.index == 52725, 'ap_hi'] = 113
train.at[train.index == 52725, 'ap_lo'] = 90
train.at[train.index == 57646, 'ap_hi'] = 90
train.at[train.index == 57646, 'ap_lo'] = 30
train.at[train.index == 58349, 'ap_hi'] = 80
train.at[train.index == 58349, 'ap_lo'] = 40
train.at[train.index == 58374, 'ap_hi'] = 160
train.at[train.index == 58374, 'ap_lo'] = 80
train.at[train.index == 58728, 'ap_hi'] = 120
train.at[train.index == 58728, 'ap_lo'] = 80
train.at[train.index == 59301, 'ap_hi'] = 80
train.at[train.index == 59301, 'ap_lo'] = 60
train.at[train.index == 60565, 'ap_hi'] = 90
train.at[train.index == 60565, 'ap_lo'] = 60
train.at[train.index == 60948, 'ap_hi'] = 90
train.at[train.index == 60948, 'ap_lo'] = 60
train.at[train.index == 61618, 'ap_hi'] = 140
train.at[train.index == 61618, 'ap_lo'] = 90
train.at[train.index == 61725, 'ap_hi'] = 140
train.at[train.index == 61725, 'ap_lo'] = 80
train.at[train.index == 62154, 'ap_hi'] = 130
train.at[train.index == 62154, 'ap_lo'] = 80
train.at[train.index == 66998, 'ap_hi'] = 140
train.at[train.index == 66998, 'ap_lo'] = 90
train.at[train.index == 67502, 'ap_hi'] = 140
train.at[train.index == 67502, 'ap_lo'] = 90
train.at[train.index == 69672, 'ap_hi'] = 140
train.at[train.index == 69672, 'ap_lo'] = 90
train.at[train.index == 72539, 'ap_hi'] = 140
train.at[train.index == 72539, 'ap_lo'] = 90
train.at[train.index == 73356, 'ap_hi'] = 110
train.at[train.index == 73356, 'ap_lo'] = 80
train.at[train.index == 77010, 'ap_hi'] = 90
train.at[train.index == 77010, 'ap_lo'] = 60
train.at[train.index == 79116, 'ap_hi'] = 130
train.at[train.index == 79116, 'ap_lo'] = 80
train.at[train.index == 79679, 'ap_hi'] = 130
train.at[train.index == 79679, 'ap_lo'] = 80
train.at[train.index == 81769, 'ap_hi'] = 130
train.at[train.index == 81769, 'ap_lo'] = 90
train.at[train.index == 82660, 'ap_hi'] = 110
train.at[train.index == 82660, 'ap_lo'] = 80
train.at[train.index == 91364, 'ap_hi'] = 120
train.at[train.index == 91364, 'ap_lo'] = 90
train.at[train.index == 92655, 'ap_hi'] = 90
train.at[train.index == 92655, 'ap_lo'] = 60
train.at[train.index == 98095, 'ap_hi'] = 90
train.at[train.index == 98095, 'ap_lo'] = 60
train.at[train.index == 99089, 'ap_hi'] = 200
train.at[train.index == 99089, 'ap_lo'] = 100

train.loc[(train['ap_lo'] > 5) & (train['ap_lo'] < 10), 'ap_lo'] *= 10
train.loc[train['ap_lo'] == 0, 'ap_lo'] = 100
train.loc[train['ap_lo'] == 10, 'ap_lo'] *= 10
train.loc[train['ap_lo'] == 1, 'ap_lo'] *= 100
train.loc[train['ap_lo'] == -70, 'ap_lo'] = 70
train.loc[train['ap_lo'] == 20, 'ap_lo'] = 80
train.loc[train['ap_lo'] == 15, 'ap_lo'] = 45

train.at[train['ap_lo'] == 1000, 'ap_lo'] = 100
train.at[train['ap_lo'] == 1100, 'ap_lo'] = 110
train.at[train['ap_lo'] == 1200, 'ap_lo'] = 120

train.at[train['ap_lo'] == 10000, 'ap_lo'] = 100
train.at[train['ap_lo'] == 8000, 'ap_lo'] = 80
train.at[train['ap_lo'] == 800, 'ap_lo'] = 80
train.at[(train['ap_lo'] > 800) & (train['ap_lo'] < 810), 'ap_lo'] = 80
train.at[(train['ap_lo'] > 1000) & (train['ap_lo'] < 1100), 'ap_lo'] = 100
train.at[(train['ap_lo'] > 9000) & (train['ap_lo'] < 9101), 'ap_lo'] = 90
train.at[train.index == 4208, 'ap_hi'] = 140
train.at[train.index == 4208, 'ap_lo'] = 80
train.at[train.index == 6992, 'ap_hi'] = 170
train.at[train.index == 6992, 'ap_lo'] = 80
train.at[(train['ap_lo'] > 900) & (train['ap_lo'] < 910), 'ap_lo'] = 90
train.at[(train['ap_lo'] > 800) & (train['ap_lo'] < 870), 'ap_lo'] = 80
train.at[train.index == 14006, 'ap_hi'] = 160
train.at[train.index == 14006, 'ap_lo'] = 90
train.at[(train['ap_lo'] > 700) & (train['ap_lo'] < 770), 'ap_lo'] = 70
train.loc[(train['ap_lo'] > 1100) & (train['ap_lo'] < 1200), 'ap_lo'] /= 10
train.at[train.index == 20438, 'ap_hi'] = 160
train.at[train.index == 20438, 'ap_lo'] = 70
train.at[train.index == 22180, 'ap_hi'] = 170
train.at[train.index == 22180, 'ap_lo'] = 95
train.loc[(train['ap_lo'] > 8000) & (train['ap_lo'] < 8900), 'ap_lo'] /= 100
train.at[train.index == 28449, 'ap_hi'] = 180
train.at[train.index == 28449, 'ap_lo'] = 150
train.at[train.index == 33191, 'ap_hi'] = 112
train.at[train.index == 33191, 'ap_lo'] = 57
train.at[train.index == 37746, 'ap_hi'] = 170
train.at[train.index == 37746, 'ap_lo'] = 110
train.at[train.index == 42591, 'ap_hi'] = 190
train.at[train.index == 42591, 'ap_lo'] = 140
train.at[train.index == 44592, 'ap_hi'] = 170
train.at[train.index == 44592, 'ap_lo'] = 95
train.at[train.index == 53070, 'ap_hi'] = 180
train.at[train.index == 53070, 'ap_lo'] = 150
train.at[train.index == 59157, 'ap_hi'] = 150
train.at[train.index == 59157, 'ap_lo'] = 109
train.at[train.index == 61901, 'ap_hi'] = 200
train.at[train.index == 61901, 'ap_lo'] = 110
train.at[train.index == 62058, 'ap_hi'] = 130
train.at[train.index == 62058, 'ap_lo'] = 98
train.at[train.index == 62938, 'ap_hi'] = 160
train.at[train.index == 62938, 'ap_lo'] = 90
train.at[train.index == 68825, 'ap_hi'] = 180
train.at[train.index == 68825, 'ap_lo'] = 95
train.at[train.index == 70263, 'ap_hi'] = 200
train.at[train.index == 70263, 'ap_lo'] = 140
train.at[train.index == 70400, 'ap_hi'] = 190
train.at[train.index == 70400, 'ap_lo'] = 170
train.at[train.index == 71695, 'ap_hi'] = 140
train.at[train.index == 71695, 'ap_lo'] = 90
train.at[train.index == 75482, 'ap_hi'] = 125
train.at[train.index == 75482, 'ap_lo'] = 68
train.at[train.index == 75520, 'ap_hi'] = 160
train.at[train.index == 75520, 'ap_lo'] = 90
train.at[train.index == 78873, 'ap_hi'] = 130
train.at[train.index == 78873, 'ap_lo'] = 90
train.at[train.index == 78905, 'ap_hi'] = 160
train.at[train.index == 78905, 'ap_lo'] = 100
train.at[train.index == 80714, 'ap_hi'] = 130
train.at[train.index == 80714, 'ap_lo'] = 70
train.at[train.index == 81683, 'ap_hi'] = 140
train.at[train.index == 81683, 'ap_lo'] = 90
train.at[train.index == 84860, 'ap_hi'] = 176
train.at[train.index == 84860, 'ap_lo'] = 120
train.at[train.index == 88752, 'ap_hi'] = 140
train.at[train.index == 88752, 'ap_lo'] = 90
train.at[train.index == 91264, 'ap_hi'] = 130
train.at[train.index == 91264, 'ap_lo'] = 90
train.at[train.index == 91638, 'ap_hi'] = 160
train.at[train.index == 91638, 'ap_lo'] = 95
train.at[train.index == 91794, 'ap_hi'] = 125
train.at[train.index == 91794, 'ap_lo'] = 88
train.at[train.index == 91842, 'ap_hi'] = 120
train.at[train.index == 91842, 'ap_lo'] = 87
train.at[train.index == 94377, 'ap_hi'] = 220
train.at[train.index == 94377, 'ap_lo'] = 112
train.at[train.index == 94387, 'ap_hi'] = 160
train.at[train.index == 94387, 'ap_lo'] = 95
train.at[train.index == 95886, 'ap_hi'] = 113
train.at[train.index == 95886, 'ap_lo'] = 57
train.at[train.index == 99006, 'ap_hi'] = 120
train.at[train.index == 99006, 'ap_lo'] = 60
train.at[train.index == 91933, 'ap_hi'] = 130
train.at[train.index == 91933, 'ap_lo'] = 60
train.at[train.index == 94673, 'ap_hi'] = 160
train.at[train.index == 94673, 'ap_lo'] = 100

train['ap_diff'] = train['ap_hi'] - train['ap_lo']

ap_hi = list(train[train['ap_diff'] < 0]['ap_hi'].values)
ap_lo = list(train[train['ap_diff'] < 0]['ap_lo'].values)

train.loc[train['ap_diff'] < 0, 'ap_hi'] = ap_lo
train.loc[train['ap_diff'] < 0, 'ap_lo'] = ap_hi
print('After AP     clearance: ', train.shape)

train_cholesterol = pd.get_dummies(train['cholesterol'], prefix='ch')
train = pd.DataFrame(pd.concat([train, train_cholesterol], axis=1))
# del train['cholesterol']
train_gluc = pd.get_dummies(train['gluc'], prefix='gl')
train = pd.DataFrame(pd.concat([train, train_gluc], axis=1))
# del train['gluc']

""" test  """
test['age_y'] = np.floor(test['age'] / 365.25)
test['gender'] -= 1

test.at[test['height'] == 110, 'height'] = 170
test.at[test['height'] == 58, 'height'] = 158
test.at[test['height'] == 116, 'height'] = 176
test.at[test['height'] == 65, 'height'] = 165
test.at[test['height'] == 100, 'height'] = 170
test.at[test['height'] == 68, 'height'] = 168
test.at[test['height'] == 72, 'height'] = 172
test.at[test['height'] == 114, 'height'] = 174
test.at[test['height'] == 50, 'height'] = 150
test.at[test['height'] == 62, 'height'] = 162
test.at[test['height'] == 60, 'height'] = 160
test.at[test['height'] == 105, 'height'] = 185
test.at[test['height'] == 87, 'height'] = 187
test.at[test['height'] == 119, 'height'] = 179
test.at[test['height'] == 52, 'height'] = 152
test.at[test['height'] == 56, 'height'] = 156
test.at[test['height'] == 102, 'height'] = 172

test.at[test['weight'] == 13, 'weight'] = 73
test.at[test['weight'] == 10, 'weight'] = 70
test.at[test['weight'] == 30, 'weight'] = 80
test.at[test['weight'] == 16.3, 'weight'] = 130
test.at[test['weight'] == 20, 'weight'] = 120
test.loc[(test['weight'] > 30) & (test['weight'] < 40), 'weight'] += 50

test.at[test['ap_hi'] == -130, 'ap_hi'] = 130
test.at[test['ap_hi'] == -12, 'ap_hi'] = 120
test.at[test.index == 6580, 'ap_hi'] = 110
test.at[test.index == 6580, 'ap_lo'] = 99
test.at[test.index == 42755, 'ap_hi'] = 100
test.at[test.index == 42755, 'ap_lo'] = 30
test.at[test.index == 51749, 'ap_hi'] = 120
test.at[test.index == 51749, 'ap_lo'] = 88
test.at[test.index == 5685, 'ap_hi'] = 100
test.at[test.index == 7465, 'ap_hi'] = 120
test.at[test.index == 7465, 'ap_lo'] = 110
test.at[test.index == 12641, 'ap_hi'] = 110
test.at[test.index == 12641, 'ap_lo'] = 70
test.at[test.index == 18180, 'ap_hi'] = 120
test.at[test.index == 18180, 'ap_lo'] = 110
test.at[test.index == 22401, 'ap_hi'] = 110
test.at[test.index == 22401, 'ap_lo'] = 70
test.at[test.index == 28251, 'ap_hi'] = 110
test.at[test.index == 28251, 'ap_lo'] = 70
test.at[test.index == 43735, 'ap_hi'] = 100
test.at[test.index == 43735, 'ap_lo'] = 70
test.at[test.index == 45374, 'ap_hi'] = 110
test.at[test.index == 45374, 'ap_lo'] = 80
test.at[test.index == 51147, 'ap_hi'] = 110
test.at[test.index == 51147, 'ap_lo'] = 80
test.at[test.index == 54108, 'ap_hi'] = 110
test.at[test.index == 54108, 'ap_lo'] = 70
test.at[test.index == 65117, 'ap_hi'] = 110
test.at[test.index == 65117, 'ap_lo'] = 70
test.at[test.index == 79396, 'ap_hi'] = 100
test.at[test.index == 79396, 'ap_lo'] = 60
test.at[test.index == 93377, 'ap_hi'] = 110
test.at[test.index == 93377, 'ap_lo'] = 70
test.at[test.index == 96170, 'ap_hi'] = 110
test.at[test.index == 96170, 'ap_lo'] = 80
test.at[test.index == 97818, 'ap_hi'] = 115
test.at[test.index == 97818, 'ap_lo'] = 70
test.at[test.index == 975, 'ap_hi'] = 120
test.at[test.index == 18219, 'ap_hi'] = 120
test.at[test.index == 22105, 'ap_hi'] = 120
test.at[test.index == 24062, 'ap_hi'] = 120
test.at[test.index == 25213, 'ap_hi'] = 120
test.at[test.index == 27360, 'ap_hi'] = 120
test.at[test.index == 27770, 'ap_hi'] = 120
test.at[test.index == 28690, 'ap_hi'] = 120
test.at[test.index == 33417, 'ap_hi'] = 120
test.at[test.index == 34199, 'ap_hi'] = 120
test.at[test.index == 36899, 'ap_hi'] = 120
test.at[test.index == 38276, 'ap_hi'] = 120
test.at[test.index == 49115, 'ap_hi'] = 120
test.at[test.index == 53263, 'ap_hi'] = 120
test.at[test.index == 54323, 'ap_hi'] = 120
test.at[test.index == 55188, 'ap_hi'] = 120
test.at[test.index == 63286, 'ap_hi'] = 120
test.at[test.index == 64376, 'ap_hi'] = 120
test.at[test.index == 71769, 'ap_hi'] = 120
test.at[test.index == 74693, 'ap_hi'] = 120
test.at[test.index == 77933, 'ap_hi'] = 120
test.at[test.index == 79220, 'ap_hi'] = 120
test.at[test.index == 82656, 'ap_hi'] = 120
test.at[test.index == 82681, 'ap_hi'] = 120
test.at[test.index == 83267, 'ap_hi'] = 120
test.at[test.index == 88480, 'ap_hi'] = 120
test.at[test.index == 91218, 'ap_hi'] = 120
test.at[test.index == 97128, 'ap_hi'] = 120
test.at[test.index == 97639, 'ap_hi'] = 120
test.at[test.index == 99357, 'ap_hi'] = 120
test.at[test.index == 1706, 'ap_hi'] = 130
test.at[test.index == 6817, 'ap_hi'] = 130
test.at[test.index == 6823, 'ap_hi'] = 130
test.at[test.index == 35587, 'ap_hi'] = 130
test.at[test.index == 38604, 'ap_hi'] = 130
test.at[test.index == 51483, 'ap_hi'] = 130
test.at[test.index == 51490, 'ap_hi'] = 130
test.at[test.index == 57170, 'ap_hi'] = 130
test.at[test.index == 59180, 'ap_hi'] = 130
test.at[test.index == 60427, 'ap_hi'] = 130
test.at[test.index == 81860, 'ap_hi'] = 130
test.at[test.index == 88306, 'ap_hi'] = 130
test.at[test.index == 13452, 'ap_hi'] = 140
test.at[test.index == 25586, 'ap_hi'] = 140
test.at[test.index == 34481, 'ap_hi'] = 140
test.at[test.index == 37680, 'ap_hi'] = 140
test.at[test.index == 38139, 'ap_hi'] = 140
test.at[test.index == 92087, 'ap_hi'] = 140
test.at[test.index == 34568, 'ap_hi'] = 150
test.at[test.index == 40877, 'ap_hi'] = 150
test.at[test.index == 78923, 'ap_hi'] = 150
test.at[test.index == 82519, 'ap_hi'] = 150
test.at[test.index == 85370, 'ap_hi'] = 150
test.at[test.index == 8272, 'ap_hi'] = 160
test.at[test.index == 21287, 'ap_hi'] = 160
test.at[test.index == 83595, 'ap_hi'] = 160
test.at[test.index == 27649, 'ap_hi'] = 170
test.at[test.index == 27649, 'ap_lo'] = 120
test.at[test.index == 47235, 'ap_hi'] = 170
test.at[test.index == 86569, 'ap_hi'] = 170
test.at[test.index == 65738, 'ap_hi'] = 190
test.at[test.index == 9856, 'ap_hi'] = 120
test.at[test.index == 1079, 'ap_hi'] = 140
test.at[test.index == 1079, 'ap_lo'] = 60
test.at[test.index == 5736, 'ap_hi'] = 107
test.at[test.index == 5736, 'ap_lo'] = 70
test.at[test.index == 5806, 'ap_hi'] = 220
test.at[test.index == 5806, 'ap_lo'] = 130
test.at[test.index == 17266, 'ap_hi'] = 120
test.at[test.index == 17266, 'ap_lo'] = 80
test.at[test.index == 23199, 'ap_hi'] = 157
test.at[test.index == 23199, 'ap_lo'] = 70
test.at[test.index == 24871, 'ap_hi'] = 106
test.at[test.index == 24871, 'ap_lo'] = 60
test.at[test.index == 25868, 'ap_hi'] = 140
test.at[test.index == 25868, 'ap_lo'] = 90
test.at[test.index == 29568, 'ap_hi'] = 160
test.at[test.index == 29568, 'ap_lo'] = 70
test.at[test.index == 30877, 'ap_hi'] = 106
test.at[test.index == 30877, 'ap_lo'] = 60
test.at[test.index == 31334, 'ap_hi'] = 140
test.at[test.index == 31334, 'ap_lo'] = 90
test.at[test.index == 35256, 'ap_hi'] = 150
test.at[test.index == 35256, 'ap_lo'] = 80
test.at[test.index == 44904, 'ap_hi'] = 140
test.at[test.index == 44904, 'ap_lo'] = 90
test.at[test.index == 45258, 'ap_hi'] = 140
test.at[test.index == 45258, 'ap_lo'] = 80
test.at[test.index == 48185, 'ap_hi'] = 140
test.at[test.index == 48185, 'ap_lo'] = 90
test.at[test.index == 50789, 'ap_hi'] = 140
test.at[test.index == 50789, 'ap_lo'] = 100
test.at[test.index == 51573, 'ap_hi'] = 140
test.at[test.index == 51573, 'ap_lo'] = 80
test.at[test.index == 56466, 'ap_hi'] = 140
test.at[test.index == 56466, 'ap_lo'] = 90
test.at[test.index == 61818, 'ap_hi'] = 160
test.at[test.index == 61818, 'ap_lo'] = 80
test.at[test.index == 62837, 'ap_hi'] = 150
test.at[test.index == 62837, 'ap_lo'] = 90
test.at[test.index == 64479, 'ap_hi'] = 113
test.at[test.index == 64479, 'ap_lo'] = 95
test.at[test.index == 81470, 'ap_hi'] = 120
test.at[test.index == 81470, 'ap_lo'] = 80
test.at[test.index == 86863, 'ap_hi'] = 210
test.at[test.index == 86863, 'ap_lo'] = 100
test.at[test.index == 88161, 'ap_hi'] = 106
test.at[test.index == 88161, 'ap_lo'] = 60
test.at[test.index == 96853, 'ap_hi'] = 110
test.at[test.index == 96853, 'ap_lo'] = 80
test.at[test.index == 99645, 'ap_hi'] = 210
test.at[test.index == 99645, 'ap_lo'] = 100
test.at[test.index == 99929, 'ap_hi'] = 90
test.at[test.index == 99929, 'ap_lo'] = 60

test.at[test.index == 80604, 'ap_lo'] = 90
test.at[test.index == 8272, 'ap_lo'] = 100
test.at[test.index == 21287, 'ap_lo'] = 100
test.at[test.index == 22925, 'ap_lo'] = 70
test.at[test.index == 25442, 'ap_lo'] = 100
test.at[test.index == 26367, 'ap_lo'] = 90
test.at[test.index == 36953, 'ap_lo'] = 100
test.at[test.index == 51515, 'ap_lo'] = 90
test.at[test.index == 53405, 'ap_lo'] = 60
test.at[test.index == 56916, 'ap_lo'] = 80
test.at[test.index == 57993, 'ap_lo'] = 90
test.at[test.index == 66969, 'ap_lo'] = 80
test.at[test.index == 81975, 'ap_lo'] = 80
test.at[test.index == 89684, 'ap_lo'] = 70
test.at[test.index == 90606, 'ap_lo'] = 90
test.at[test.index == 97796, 'ap_lo'] = 80
test.at[test['ap_lo'] == 30, 'ap_lo'] = 60
test.at[test['ap_lo'] == 20, 'ap_lo'] = 60
test.at[test['ap_lo'] == 1000, 'ap_lo'] = 100
test.at[test['ap_lo'] == 1100, 'ap_lo'] = 110
test.at[test['ap_lo'] == 1200, 'ap_lo'] = 120
test.at[test['ap_lo'] == 900, 'ap_lo'] = 90
test.at[test['ap_lo'] == 800, 'ap_lo'] = 80
test.at[test['ap_lo'] == 1300, 'ap_lo'] = 130
test.at[test['ap_lo'] == 1110, 'ap_lo'] = 110
test.at[test['ap_lo'] == 1120, 'ap_lo'] = 112
test.at[test['ap_lo'] == 1110, 'ap_lo'] = 110
test.at[test['ap_lo'] == 4100, 'ap_lo'] = 140
test.at[test['ap_lo'] == 1099, 'ap_lo'] = 109
test.at[test['ap_lo'] == 190, 'ap_lo'] = 110
test.at[test['ap_lo'] == 808, 'ap_lo'] = 80
test.at[test['ap_lo'] == 1003, 'ap_lo'] = 100
test.at[test['ap_lo'] == 910, 'ap_lo'] = 90
test.at[test['ap_lo'] == 8099, 'ap_lo'] = 90
test.at[test['ap_lo'] == 470, 'ap_lo'] = 70
test.at[test['ap_lo'] == 8100, 'ap_lo'] = 80
test.at[test['ap_lo'] == 9100, 'ap_lo'] = 90
test.at[test['ap_lo'] == 4700, 'ap_lo'] = 70
test.at[test['ap_lo'] == 801, 'ap_lo'] = 80
test.at[test['ap_lo'] == 708, 'ap_lo'] = 70
test.at[test['ap_lo'] == 1011, 'ap_lo'] = 110
test.at[test['ap_lo'] == 1004, 'ap_lo'] = 100
test.at[test['ap_lo'] == 701, 'ap_lo'] = 70
test.at[test['ap_lo'] == 1001, 'ap_lo'] = 100
test.at[test['ap_lo'] == 880, 'ap_lo'] = 88
test.at[test['ap_lo'] == 8022, 'ap_lo'] = 80
test.at[test['ap_lo'] == 809, 'ap_lo'] = 80
test.at[test['ap_lo'] == 1066, 'ap_lo'] = 106
test.at[test['ap_lo'] == 1101, 'ap_lo'] = 110
test.at[test['ap_lo'] == 809, 'ap_lo'] = 80
test.at[test['ap_lo'] == 1009, 'ap_lo'] = 100
test.at[test.index == 2080, 'ap_hi'] = 130
test.at[test.index == 2080, 'ap_lo'] = 90
test.at[test.index == 3777, 'ap_hi'] = 160
test.at[test.index == 3777, 'ap_lo'] = 90
test.at[test.index == 3825, 'ap_hi'] = 130
test.at[test.index == 3825, 'ap_lo'] = 80
test.at[test.index == 4345, 'ap_hi'] = 150
test.at[test.index == 4345, 'ap_lo'] = 90
test.at[test.index == 4933, 'ap_hi'] = 130
test.at[test.index == 4933, 'ap_lo'] = 90
test.at[test.index == 6832, 'ap_hi'] = 120
test.at[test.index == 6832, 'ap_lo'] = 80
test.at[test.index == 10591, 'ap_hi'] = 160
test.at[test.index == 10591, 'ap_lo'] = 100
test.at[test.index == 10815, 'ap_hi'] = 120
test.at[test.index == 10815, 'ap_lo'] = 80
test.at[test.index == 11963, 'ap_hi'] = 150
test.at[test.index == 11963, 'ap_lo'] = 90
test.at[test.index == 14623, 'ap_hi'] = 120
test.at[test.index == 14623, 'ap_lo'] = 80
test.at[test.index == 15128, 'ap_hi'] = 120
test.at[test.index == 15128, 'ap_lo'] = 80
test.at[test.index == 21949, 'ap_hi'] = 120
test.at[test.index == 21949, 'ap_lo'] = 80
test.at[test.index == 24054, 'ap_hi'] = 170
test.at[test.index == 24054, 'ap_lo'] = 90
test.at[test.index == 26749, 'ap_hi'] = 130
test.at[test.index == 26749, 'ap_lo'] = 80
test.at[test.index == 27402, 'ap_hi'] = 100
test.at[test.index == 27402, 'ap_lo'] = 60
test.at[test.index == 27569, 'ap_hi'] = 110
test.at[test.index == 27569, 'ap_lo'] = 60
test.at[test.index == 33552, 'ap_hi'] = 110
test.at[test.index == 33552, 'ap_lo'] = 70
test.at[test.index == 34871, 'ap_hi'] = 120
test.at[test.index == 34871, 'ap_lo'] = 80
test.at[test.index == 37329, 'ap_hi'] = 150
test.at[test.index == 37329, 'ap_lo'] = 95
test.at[test.index == 37445, 'ap_hi'] = 120
test.at[test.index == 37445, 'ap_lo'] = 70
test.at[test.index == 45254, 'ap_hi'] = 100
test.at[test.index == 45254, 'ap_lo'] = 60
test.at[test.index == 53710, 'ap_hi'] = 110
test.at[test.index == 53710, 'ap_lo'] = 70
test.at[test.index == 54795, 'ap_hi'] = 120
test.at[test.index == 54795, 'ap_lo'] = 80
test.at[test.index == 57548, 'ap_hi'] = 120
test.at[test.index == 57548, 'ap_lo'] = 80
test.at[test.index == 57702, 'ap_hi'] = 115
test.at[test.index == 57702, 'ap_lo'] = 80
test.at[test.index == 60111, 'ap_hi'] = 120
test.at[test.index == 60111, 'ap_lo'] = 80
test.at[test.index == 67443, 'ap_hi'] = 120
test.at[test.index == 67443, 'ap_lo'] = 80
test.at[test.index == 74101, 'ap_hi'] = 130
test.at[test.index == 74101, 'ap_lo'] = 95
test.at[test.index == 75103, 'ap_hi'] = 120
test.at[test.index == 75103, 'ap_lo'] = 80
test.at[test.index == 75960, 'ap_hi'] = 160
test.at[test.index == 75960, 'ap_lo'] = 90
test.at[test.index == 76248, 'ap_hi'] = 120
test.at[test.index == 76248, 'ap_lo'] = 80
test.at[test.index == 78523, 'ap_hi'] = 120
test.at[test.index == 78523, 'ap_lo'] = 80
test.at[test.index == 82483, 'ap_hi'] = 120
test.at[test.index == 82483, 'ap_lo'] = 80
test.at[test.index == 83193, 'ap_hi'] = 120
test.at[test.index == 83193, 'ap_lo'] = 80
test.at[test.index == 86627, 'ap_hi'] = 150
test.at[test.index == 86627, 'ap_lo'] = 90
test.at[test.index == 88055, 'ap_hi'] = 110
test.at[test.index == 88055, 'ap_lo'] = 60
test.at[test.index == 88522, 'ap_hi'] = 170
test.at[test.index == 88522, 'ap_lo'] = 95
test.at[test.index == 88602, 'ap_hi'] = 110
test.at[test.index == 88602, 'ap_lo'] = 70
test.at[test.index == 89839, 'ap_hi'] = 140
test.at[test.index == 89839, 'ap_lo'] = 90
test.at[test.index == 91594, 'ap_hi'] = 110
test.at[test.index == 91594, 'ap_lo'] = 70
test.at[test.index == 93758, 'ap_hi'] = 140
test.at[test.index == 93758, 'ap_lo'] = 80
test.at[test.index == 94426, 'ap_hi'] = 120
test.at[test.index == 94426, 'ap_lo'] = 110
test.at[test.index == 97779, 'ap_hi'] = 120
test.at[test.index == 97779, 'ap_lo'] = 80
test.at[test.index == 99068, 'ap_hi'] = 140
test.at[test.index == 99068, 'ap_lo'] = 90
test.at[test.index == 99353, 'ap_hi'] = 120
test.at[test.index == 99353, 'ap_lo'] = 80

test['ap_diff'] = test['ap_hi'] - test['ap_lo']

test_cholesterol = pd.get_dummies(test['cholesterol'], prefix='ch')
test = pd.DataFrame(pd.concat([test, test_cholesterol], axis=1))
# del test['cholesterol']
test_gluc = pd.get_dummies(test['gluc'], prefix='gl')
test = pd.DataFrame(pd.concat([test, test_gluc], axis=1))
# del test['gluc']

test['smoke'] = test['smoke'].replace('None', 0)
test['alco'] = test['alco'].replace('None', 0)
test['active'] = test['active'].replace('None', 1)

"""
 END set transformation
"""
test['ap_lo'].plot.hist(100)

plt.figure(figsize=(8, 6))
plt.scatter(range(test.shape[0]), np.sort(test['ap_lo'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('var', fontsize=12)
plt.show()

train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
