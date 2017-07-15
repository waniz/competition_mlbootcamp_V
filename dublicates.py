import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from FEATURES import AddFeatures

pd.set_option('display.max_columns', 40)
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

train = pd.DataFrame(add_features.train)
test = pd.DataFrame(add_features.test)

# """ check """
# print(test[(test['gender'] == 0) & (test['height'] == 165) & (test['weight'] == 68) & (test['ap_hi'] == 120) &
#            (test['ap_lo'] == 80) & (test['cholesterol'] == 1) & (test['gluc'] == 1) & (test['smoke'] == 0) &
#            (test['alco'] == 0) & (test['active'] == 1) & (test['age_y'] == 49)])
#
# train_df = train[train.duplicated(subset=['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
#                                           'smoke', 'alco', 'active', 'age_y'], keep=False)]
# train.drop_duplicates(subset=['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
#                               'smoke', 'alco', 'active', 'age_y'], inplace=True)
# print('Dublicates train', train_df.shape)
#
# test_df = test[test.duplicated(subset=['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
#                                        'smoke', 'alco', 'active', 'age_y'], keep=False)]
# test.drop_duplicates(subset=['gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc',
#                              'smoke', 'alco', 'active', 'age_y'], inplace=True)
# print('Dublicates test', test_df.shape)
#
# print(test_df[:1])

print(train[(train['gender'] == 0) & (train['height'] == 165) & (train['weight'] == 68) & (train['ap_hi'] == 120) &
            (train['ap_lo'] == 80) & (train['cholesterol'] == 1) & (train['gluc'] == 1) & (train['smoke'] == 0) &
            (train['alco'] == 0) & (train['active'] == 1) & (train['age_y'] == 49)])

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train

answers = pd.read_csv('answer.csv', names=['val'])
print(answers.shape, test.shape)

answers['index'] = test.index

print(answers[answers['index'] == 93])
print(answers[answers['index'] == 33201])
print(answers[answers['index'] == 69726])

# 93 33201 69726


