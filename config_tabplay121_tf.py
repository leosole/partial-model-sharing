import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np

batch_size = 2048
all_col = range(16)
cols = [x for x in all_col]
num_columns = len(cols)
cols_1 = [6,2,10,11,0,5,9,8,7,13,4]
cols_2 = [1,2,10,11,0,5,9,8,7,12,3]
shared_columns = [col for col in cols_1 if col in cols_2]
ind_1 = [col for col in cols_1 if col not in shared_columns]
ind_2 = [col for col in cols_2 if col not in shared_columns]
client1 = {
    'end_train': 120000, 'initial_train': 0, 'initial_test': 240000, 'end_test': 300000, 'shared':shared_columns, 'ind':ind_1
}
client2 = {
    'end_train': 240000, 'initial_train': 120000, 'initial_test': 240000, 'end_test': 300000, 'shared':shared_columns, 'ind':ind_2
}
global_client = {
    'end_train': 240000, 'initial_train': 0, 'initial_test':240000, 'end_test': 300000
}
path = 'data/TabularPlaygroundJan2021.csv'
shared_layers = [128,64]
ind_layers = [32, 32]
agg_layers = [96, 32]
local_layers = [128,64,32]
server_layers = [128,64,32]
dropout = 0.3
weight_decay=0
label = -1
rounds = 100
epochs = 2
random = 2
freeze = False
resample = False
activation='linear'
targets = 1
metrics = tf.keras.metrics.RootMeanSquaredError()
# metrics = tfa.metrics.F1Score(num_classes=7)
loss = 'mse'
lr = 0.003