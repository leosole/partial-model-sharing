import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

batch_size = 128
all_col = range(23)
cols = [x for x in all_col]
num_columns = len(cols)
cols_1 = [6,2,10,11,0,5,9,8,7,13,4]
cols_2 = [1,2,10,11,0,5,9,8,7,12,3]
cols_1 = [x for x in all_col if x % 2 == 0 or x < 13]
cols_2 = [x for x in all_col if x % 2 != 0 or x < 13]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 400, 'initial_train': 0, 'initial_test': 800, 'end_test': 1000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 800, 'initial_train': 400, 'initial_test': 800, 'end_test': 1000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 800, 'initial_train': 0, 'initial_test':800, 'end_test': 1000
}
path = 'data/german_credit_numeric.csv'
shared_layers = [128,128]
ind_layers = [128,128]
agg_layers = [256,128]
local_layers = [128,128,128]
server_layers = [128,128,128]
dropout = 0.1
weight_decay=0
label = -1
rounds = 100
epochs = 2
random = 5
freeze = False
resample = False
activation='sigmoid'
targets = 1
# metrics = tf.keras.metrics.RootMeanSquaredError()
metrics = tfa.metrics.F1Score(num_classes=1)
loss = 'binary_crossentropy'
lr = 0.003