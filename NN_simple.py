import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from FEATURES import AddFeatures

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
pd.set_option('display.max_columns', 24)
pd.set_option('display.width', 1000)
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

best_columns = [
    'gender', 'weight', 'ap_hi', 'ap_lo', 'smoke', 'alco', 'active', 'age_y', 'ap_diff', 'ch_1', 'ch_2', 'ch_3', 'gl_1',
    'gl_2', 'gl_3', 'bmi', 'sist_formula', 'map', 'F_score', 'ap_log', 'ap_/'
]

Y = train['cardio'].values
train.drop('cardio', axis=1, inplace=True)
X = train[best_columns]
test = test[best_columns]

scaler = StandardScaler()
X = scaler.fit_transform(X)

scaler = StandardScaler()
test = scaler.fit_transform(test)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = Sequential()
model.add(Dense(input_dim=X.shape[1], activation='relu', output_dim=21))
model.add(Dropout(0.1))
model.add(Dense(21, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model_json = model.to_json()
with open('keras_saves/current_model.json', 'w') as json_file:
    json_file.write(model_json)
filepath = "keras_saves/best_weights.hdf5"
# filepath = "keras_saves/weights_ep_loss_{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model.fit(X, Y, nb_epoch=25, batch_size=10, verbose=2, validation_split=0.1, callbacks=[checkpoint])

model.load_weights(filepath)
score, accuracy = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
print('\nAccuracy score :', round(accuracy, 4), 'score =', score)

"""
ANSWER module
"""
print('Answer module started')
y_predict = model.predict(test)
print(y_predict)
pd.Series(y_predict[:, 0]).to_csv('answer.csv', index=False)
